"""Tune the int8 prefill GEMM launch config and compare with the fp16 path."""
import sys
import time
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CONFIGS = [
    (64, 64, 128, 4, 2),
    (128, 64, 128, 8, 2),
    (64, 128, 64, 4, 2),
    (128, 128, 64, 8, 3),
    (64, 64, 64, 4, 2),
    (128, 64, 256, 8, 2),
]


def bench(fn, n=10):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1e6


def main():
    if not torch.cuda.is_available():
        print("no device")
        return 1
    torch.cuda.set_per_process_memory_fraction(0.6)
    from zluda_ops import quant as Q
    for m, n, k in ((64, 2048, 2048), (512, 2048, 2048)):
        w = torch.randn(n, k, device="cuda", dtype=torch.float16) * 0.05
        x = torch.randn(m, k, device="cuda", dtype=torch.float16)
        wi8, sc = Q.quantize_int8(w, 128)
        w_deq = Q.dequantize_int8(wi8, sc, (n, k), 128)
        ref = x @ w_deq.t()
        print(f"--- m={m} n={n} k={k} ---", flush=True)
        for bm, bn, bk, warps, stages in CONFIGS:
            try:
                y = Q.int8_matmul(x, wi8, sc, block_m=bm, block_n=bn, block_k=bk,
                                  num_warps=warps, num_stages=stages)
                err = float((y.float() - ref.float()).abs().max())
                us = bench(lambda: Q.int8_matmul(x, wi8, sc, block_m=bm, block_n=bn,
                                                 block_k=bk, num_warps=warps,
                                                 num_stages=stages))
                print(f"  bm={bm} bn={bn} bk={bk} w={warps} s={stages}: {us:8.1f} us err={err:.3f}", flush=True)
            except Exception as exc:
                print(f"  bm={bm} bn={bn} bk={bk}: {type(exc).__name__} {str(exc)[:60]}", flush=True)
        base = bench(lambda: x @ w_deq.t())
        print(f"  fp16 dequant baseline: {base:8.1f} us", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())