"""Split-K / tl.dot GEMV kernel for M=16..32.

The existing row-per-program kernel (bench/gemv.py, `_gemv_mvec`) is
compute-bound at M=16/32: 155/78 GB/s instead of the 70-80% bandwidth target.
This script benchmarks a tl.dot based kernel with K-splitting so the fully
unrolled fp16 dot products run as small GEMMs and the K dimension is spread
over the CUs.

Usage:
  run-zluda.cmd venv\\Scripts\\python.exe bench\\gemv_m32.py --n 2048 --k 2048
  run-zluda.cmd venv\\Scripts\\python.exe bench\\gemv_m32.py --n 4096 --k 4096 --sweep
"""

import argparse
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import bench_median, save_report

PEAK_BW_GBS = 448.0


@triton.jit
def _gemv_splitk(w_ptr, x_ptr, b_ptr, y_ptr, N, M,
                 K: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                 BK: tl.constexpr, SPLIT_K: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_m = tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    k_per = K // SPLIT_K
    k_start = pid_k * k_per
    for k0 in range(k_start, k_start + k_per, BK):
        offs_k = k0 + tl.arange(0, BK)
        x = tl.load(x_ptr + offs_m[:, None] * K + offs_k[None, :],
                    mask=offs_m[:, None] < M, other=0.0)
        w = tl.load(w_ptr + offs_n[:, None] * K + offs_k[None, :],
                    mask=offs_n[:, None] < N, other=0.0)
        acc = tl.dot(x, tl.trans(w), acc)
    if pid_k == 0:
        acc += tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0)[None, :].to(tl.float32)
    out = y_ptr + offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    if SPLIT_K == 1:
        tl.store(out, acc.to(y_ptr.dtype.element_ty), mask=mask)
    else:
        tl.atomic_add(out, acc, mask=mask)


def run_splitk(w, x, bias, bn=128, bk=64, split_k=4, num_warps=4):
    n, k = w.shape
    m = x.shape[0]
    bm = 16 if m <= 16 else 32
    grid = (triton.cdiv(n, bn), split_k)
    if split_k == 1:
        y = torch.empty(m, n, device=w.device, dtype=w.dtype)
        _gemv_splitk[grid](w, x, bias, y, n, m, K=k, BM=bm, BN=bn, BK=bk,
                           SPLIT_K=1, num_warps=num_warps, num_stages=1)
        return y
    y32 = torch.zeros(m, n, device=w.device, dtype=torch.float32)
    _gemv_splitk[grid](w, x, bias, y32, n, m, K=k, BM=bm, BN=bn, BK=bk,
                       SPLIT_K=split_k, num_warps=num_warps, num_stages=1)
    return y32.to(w.dtype)


def traffic_bytes(n, k, m, elem_size, out_size=2):
    return (n * k * elem_size + m * k * elem_size + m * n * out_size
            + n * elem_size)


def measure(w, x, bias, config, typ, iters=3):
    bn, bk, split_k, warps = config
    run = lambda: run_splitk(w, x, bias, bn, bk, split_k, warps)
    y = run()
    ref = (x.float() @ w.float().t() + bias.float())
    err = (y.float() - ref).abs().max().item()
    ms = bench_median(run, rounds=3, iters=iters)
    n, k = w.shape
    m = x.shape[0]
    gbs = traffic_bytes(n, k, m, w.element_size()) / (ms / 1000.0) / 1e9
    return {"m": m, "config": list(config), "ms": ms, "gbs": gbs,
            "pct_peak": gbs / PEAK_BW_GBS * 100.0, "max_abs": err}


def measure_mvec(w, x, bias, bk, prod_f16, warps, iters=3):
    from gemv import _gemv_mvec

    n, k = w.shape
    m = x.shape[0]
    y = torch.empty(m, n, device=w.device, dtype=w.dtype)
    run = lambda: _gemv_mvec[(n,)](w, x, bias, y, n, K=k, M=m, BK=bk,
                                  PROD_F16=prod_f16, num_warps=warps,
                                  num_stages=1)
    run()
    ref = x.float() @ w.float().t() + bias.float()
    err = (y.float() - ref).abs().max().item()
    ms = bench_median(run, rounds=3, iters=iters)
    gbs = traffic_bytes(n, k, m, w.element_size()) / (ms / 1000.0) / 1e9
    return {"m": m, "config": ["mvec", bk, prod_f16, warps], "ms": ms,
            "gbs": gbs, "pct_peak": gbs / PEAK_BW_GBS * 100.0, "max_abs": err}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--k", type=int, default=4096)
    parser.add_argument("--ms", type=str, default="16,32")
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--mvec", action="store_true")
    parser.add_argument("--vram-fraction", type=float, default=0.85)
    args = parser.parse_args()

    torch.manual_seed(0)
    torch.cuda.set_per_process_memory_fraction(args.vram_fraction)
    print(f"VRAM fraction {args.vram_fraction}", flush=True)

    n, k = args.n, args.k
    ms = [int(v) for v in args.ms.split(",") if v]
    w = torch.randn(n, k, device="cuda", dtype=torch.float16) * 0.02
    bias = torch.randn(n, device="cuda", dtype=torch.float16) * 0.02

    configs = [(64, 64, 4, 4)]
    if args.sweep:
        configs = [(64, 64, 4, 4), (64, 64, 8, 4), (32, 64, 8, 4),
                   (64, 128, 8, 4), (128, 64, 8, 8), (32, 128, 8, 4)]
    rows = []
    for m in ms:
        x = torch.randn(m, k, device="cuda", dtype=torch.float16) * 0.1
        for cfg in configs:
            try:
                row = measure(w, x, bias, cfg, "splitk")
            except Exception as exc:
                print(f"m={m} cfg={cfg} failed: {type(exc).__name__}: {str(exc)[:120]}", flush=True)
                continue
            rows.append(row)
            print(f"m={row['m']:2d} bn,bk,split,warps={cfg}: {row['ms']:8.3f} ms "
                  f"{row['gbs']:7.1f} GB/s ({row['pct_peak']:5.1f}%) err={row['max_abs']:.4f}",
                  flush=True)
        # reference: existing row-per-program kernel
        try:
            from gemv import run_gemv
            ref_ms = bench_median(lambda: run_gemv(w, x, bias), rounds=3, iters=3)
            ref_gbs = traffic_bytes(n, k, m, w.element_size()) / (ref_ms / 1000.0) / 1e9
            print(f"m={m:2d} mvec reference: {ref_ms:8.3f} ms {ref_gbs:7.1f} GB/s "
                  f"({ref_gbs / PEAK_BW_GBS * 100:5.1f}%)", flush=True)
        except Exception as exc:
            print(f"mvec reference failed: {exc}", flush=True)
        if args.mvec:
            for bk, prod, warps in ((4096, True, 4), (4096, False, 4),
                                    (2048, True, 8), (4096, True, 8)):
                try:
                    row = measure_mvec(w, x, bias, bk, prod, warps)
                except Exception as exc:
                    print(f"m={m} mvec cfg failed: {exc}", flush=True)
                    continue
                rows.append(row)
                print(f"m={row['m']:2d} mvec bk={bk} prod16={prod} warps={warps}: "
                      f"{row['ms']:8.3f} ms {row['gbs']:7.1f} GB/s "
                      f"({row['pct_peak']:5.1f}%) err={row['max_abs']:.4f}", flush=True)
    save_report("gemv_m32", {"n": n, "k": k, "rows": rows})
    return 0


if __name__ == "__main__":
    sys.exit(main())
