import argparse

import torch
import triton
import triton.language as tl

from common import bench, save_report

DEV = "cuda"
PEAK_BW_GBS = 448.0


@triton.jit
def _gemv_rows(w_ptr, x_ptr, b_ptr, y_ptr, N, K: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * BN + tl.arange(0, BN)
    acc = tl.zeros((BN,), tl.float32)
    for k0 in range(0, K, BK):
        offs = k0 + tl.arange(0, BK)
        w = tl.load(w_ptr + rows[:, None] * K + offs[None, :], mask=rows[:, None] < N, other=0.0).to(tl.float32)
        x = tl.load(x_ptr + offs).to(tl.float32)
        acc += tl.sum(w * x[None, :], axis=1)
    acc += tl.load(b_ptr + rows, mask=rows < N, other=0.0).to(tl.float32)
    tl.store(y_ptr + rows, acc.to(y_ptr.dtype.element_ty), mask=rows < N)


@triton.jit
def _gemv4(w_ptr, x_ptr, b_ptr, y_ptr, N, K: tl.constexpr, BLOCK_K: tl.constexpr):
    row = tl.program_id(0)
    acc0 = tl.zeros((), tl.float32)
    acc1 = tl.zeros((), tl.float32)
    acc2 = tl.zeros((), tl.float32)
    acc3 = tl.zeros((), tl.float32)
    for k0 in range(0, K, BLOCK_K):
        offs = k0 + tl.arange(0, BLOCK_K)
        w = tl.load(w_ptr + row * K + offs).to(tl.float32)
        x0 = tl.load(x_ptr + offs).to(tl.float32)
        x1 = tl.load(x_ptr + K + offs).to(tl.float32)
        x2 = tl.load(x_ptr + 2 * K + offs).to(tl.float32)
        x3 = tl.load(x_ptr + 3 * K + offs).to(tl.float32)
        acc0 += tl.sum(w * x0, axis=0)
        acc1 += tl.sum(w * x1, axis=0)
        acc2 += tl.sum(w * x2, axis=0)
        acc3 += tl.sum(w * x3, axis=0)
    b = tl.load(b_ptr + row).to(tl.float32)
    tl.store(y_ptr + row, (acc0 + b).to(y_ptr.dtype.element_ty))
    tl.store(y_ptr + N + row, (acc1 + b).to(y_ptr.dtype.element_ty))
    tl.store(y_ptr + 2 * N + row, (acc2 + b).to(y_ptr.dtype.element_ty))
    tl.store(y_ptr + 3 * N + row, (acc3 + b).to(y_ptr.dtype.element_ty))


def run_gemv(w, x, bias):
    n, k = w.shape
    m = x.shape[0]
    y = torch.empty(m, n, device=w.device, dtype=w.dtype)
    if m == 1:
        bn, bk, warps = (2, 4096, 8) if w.dtype == torch.float16 else (2, 2048, 8)
        _gemv_rows[(triton.cdiv(n, bn),)](w, x[0], bias, y[0], n, K=k, BN=bn, BK=bk, num_warps=warps, num_stages=1)
    elif m == 4:
        _gemv4[(n,)](w, x, bias, y, n, K=k, BLOCK_K=1024, num_warps=8)
    else:
        raise ValueError(f"unsupported m={m}")
    return y


def correctness(w, x, bias):
    y = run_gemv(w, x, bias)
    ref = x.float() @ w.float().t() + bias.float()
    return (y.float() - ref).abs().max().item()


def perf(n=4096, k=4096, dtypes=(torch.float16, torch.float32)):
    report = []
    for dtype in dtypes:
        w = torch.randn(n, k, device=DEV, dtype=dtype) * 0.02
        bias = torch.randn(n, device=DEV, dtype=dtype) * 0.02
        for m in (1, 4):
            x = torch.randn(m, k, device=DEV, dtype=dtype) * 0.1
            err = correctness(w, x, bias)
            ms = min(bench(lambda: run_gemv(w, x, bias), warmup=5, iters=20) for _ in range(3))
            traffic = n * k * w.element_size() + m * k * x.element_size() + m * n * 2 + n * bias.element_size()
            gbs = traffic / (ms / 1000.0) / 1e9
            ref_ms = min(bench(lambda: torch.nn.functional.linear(x, w, bias), warmup=3, iters=10) for _ in range(2))
            ref_gbs = traffic / (ref_ms / 1000.0) / 1e9
            report.append({
                "dtype": str(dtype).split(".")[-1], "m": m, "n": n, "k": k,
                "ms": ms, "gbs": gbs, "pct_peak": gbs / PEAK_BW_GBS * 100.0,
                "torch_ms": ref_ms, "torch_gbs": ref_gbs, "max_abs": err,
            })
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--k", type=int, default=4096)
    args = parser.parse_args()
    torch.manual_seed(0)
    rows = perf(args.n, args.k)
    print(f"GEMV/skinny, N={args.n} K={args.k}")
    for r in rows:
        print(f"{r['dtype']:9s} m={r['m']}: {r['ms']:8.3f} ms {r['gbs']:7.1f} GB/s ({r['pct_peak']:5.1f}%) "
              f"err={r['max_abs']:.4f} | torch {r['torch_ms']:8.3f} ms {r['torch_gbs']:7.1f} GB/s")
    save_report("gemv", {"n": args.n, "k": args.k, "rows": rows})


if __name__ == "__main__":
    main()
