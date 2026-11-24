import argparse

import torch
import triton
import triton.language as tl

from common import bench, save_report

DEV = "cuda"
PEAK_BW_GBS = 448.0


@triton.jit
def _gemv_mvec(w_ptr, x_ptr, b_ptr, y_ptr, N, K: tl.constexpr, M: tl.constexpr,
               BK: tl.constexpr, PROD_F16: tl.constexpr):
    row = tl.program_id(0)
    mcols = tl.arange(0, M)
    acc = tl.zeros((M,), tl.float32)
    for k0 in range(0, K, BK):
        offs = k0 + tl.arange(0, BK)
        w = tl.load(w_ptr + row * K + offs)
        x = tl.load(x_ptr + mcols[:, None] * K + offs[None, :])
        if PROD_F16:
            acc += tl.sum(w[None, :] * x, axis=1).to(tl.float32)
        else:
            acc += tl.sum(w[None, :].to(tl.float32) * x.to(tl.float32), axis=1)
    acc += tl.load(b_ptr + row).to(tl.float32)
    tl.store(y_ptr + mcols * N + row, acc.to(y_ptr.dtype.element_ty))


CONFIGS = {
    1: (4096, 4, True),
    2: (4096, 4, True),
    4: (2048, 8, True),
    8: (4096, 8, True),
    16: (4096, 4, True),
    32: (4096, 4, False),
}


def run_gemv(w, x, bias):
    n, k = w.shape
    m = x.shape[0]
    y = torch.empty(m, n, device=w.device, dtype=w.dtype)
    bk, warps, prod_f16 = CONFIGS.get(m, (4096, 4, False))
    _gemv_mvec[(n,)](w, x, bias, y, n, K=k, M=m, BK=bk, PROD_F16=prod_f16,
                     num_warps=warps, num_stages=1)
    return y


def correctness(w, x, bias):
    y = run_gemv(w, x, bias)
    ref = x.float() @ w.float().t() + bias.float()
    return (y.float() - ref).abs().max().item()


def perf(n=4096, k=4096, dtypes=(torch.float16, torch.float32), ms=(1, 4, 8, 16, 32)):
    report = []
    for dtype in dtypes:
        w = torch.randn(n, k, device=DEV, dtype=dtype) * 0.02
        bias = torch.randn(n, device=DEV, dtype=dtype) * 0.02
        for m in ms:
            x = torch.randn(m, k, device=DEV, dtype=dtype) * 0.1
            err = correctness(w, x, bias)
            ms_time = min(bench(lambda: run_gemv(w, x, bias), warmup=5, iters=20) for _ in range(2))
            traffic = n * k * w.element_size() + m * k * x.element_size() + m * n * 2 + n * bias.element_size()
            gbs = traffic / (ms_time / 1000.0) / 1e9
            ref_ms = min(bench(lambda: torch.nn.functional.linear(x, w, bias), warmup=3, iters=10) for _ in range(2))
            ref_gbs = traffic / (ref_ms / 1000.0) / 1e9
            report.append({
                "dtype": str(dtype).split(".")[-1], "m": m, "n": n, "k": k,
                "ms": ms_time, "gbs": gbs, "pct_peak": gbs / PEAK_BW_GBS * 100.0,
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
        print(f"{r['dtype']:9s} m={r['m']:2d}: {r['ms']:8.3f} ms {r['gbs']:7.1f} GB/s ({r['pct_peak']:5.1f}%) "
              f"err={r['max_abs']:.4f} | torch {r['torch_ms']:8.3f} ms {r['torch_gbs']:7.1f} GB/s")
    save_report("gemv", {"n": args.n, "k": args.k, "rows": rows})


if __name__ == "__main__":
    main()
