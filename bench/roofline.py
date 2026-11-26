"""Roofline harness for the Phase-2 decode kernels (ZLUDA/RDNA1).

Measures each bandwidth-bound kernel and reports the achieved GB/s as a share of
the RDNA1 peak (448 GB/s) so it is obvious which kernels still have headroom.
Run under run-zluda.cmd:

    venv\\Scripts\\python.exe bench\\roofline.py
"""

import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
PEAK_GBS = 448.0
ROWS = []


def bench(fn, n=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n


def row(name, fn, nbytes, flops=0, iters=20):
    us = bench(fn, iters) * 1e6
    gbs = nbytes / (us * 1e-6) / 1e9
    tflops = flops / (us * 1e-6) / 1e12
    ROWS.append({"name": name, "us": round(us, 1), "gbs": round(gbs, 1),
                 "pct_peak": round(gbs / PEAK_GBS * 100, 1),
                 "tflops": round(tflops, 3)})
    print(f"{name:32s} {us:9.1f} us  {gbs:6.1f} GB/s ({gbs / PEAK_GBS * 100:5.1f}% peak)"
          + (f"  {tflops:6.2f} TF" if flops else ""), flush=True)


def main():
    if not torch.cuda.is_available():
        print("CUDA/ZLUDA device is not available")
        return 1
    torch.cuda.set_per_process_memory_fraction(0.7)
    torch.manual_seed(0)
    import zluda_ops.auto
    from zluda_ops import gemv as G, fused as F
    from zluda_ops.gqa_decode import gqa_decode_attn
    from zluda_ops.qkv_fused import qkv_rope_append
    from zluda_ops.embedding import embedding_lookup
    from zluda_ops.quant import int8_matmul, quantize_int8
    from zluda_ops import flash_attention as FA

    # copy / fill
    n = 128 * 1024 * 1024 // 2
    a = torch.empty(n, device="cuda", dtype=torch.float16)
    b = torch.empty_like(a)
    row("copy 128MB (D2D)", lambda: b.copy_(a), 2 * a.numel() * 2)
    row("fill 128MB", lambda: a.fill_(1.0), a.numel() * 2)

    # decode GEMVs (memory-bound: weights + output)
    for N_, K_ in ((896, 896), (4864, 896), (896, 4864), (151936, 896)):
        w = torch.randn(N_, K_, device="cuda", dtype=torch.float16)
        x = torch.randn(1, K_, device="cuda", dtype=torch.float16)
        nb = N_ * K_ * 2 + N_ * 2
        row(f"gemv {N_}x{K_}", lambda: G.linear(x, w, None, accurate=True), nb,
            flops=2 * N_ * K_)

    # GQA decode (KV + q + out traffic)
    q = torch.randn(1, 14, 1, 64, device="cuda", dtype=torch.float16)
    k = torch.randn(1, 2, 2048, 64, device="cuda", dtype=torch.float16)
    v = torch.randn(1, 2, 2048, 64, device="cuda", dtype=torch.float16)
    nb = 2 * 2 * 2048 * 64 * 2 + q.numel() * 2 + q.numel() * 2
    row("gqa_decode n=2048", lambda: gqa_decode_attn(q, k, v), nb,
        flops=4 * 2048 * 64 * 14)

    # fused residual+rmsnorm (1 token)
    x = torch.randn(1, 896, device="cuda", dtype=torch.float16)
    d = torch.randn(1, 896, device="cuda", dtype=torch.float16)
    w = torch.randn(896, device="cuda", dtype=torch.float16)
    row("add_rms_norm 896", lambda: F.add_rms_norm(x, d, w), 3 * 896 * 2, 4 * 896)

    # fused qkv+rope+append
    wqkv = torch.randn(1152, 896, device="cuda", dtype=torch.float16)
    x1 = torch.randn(896, device="cuda", dtype=torch.float16)
    cos = torch.randn(32, device="cuda", dtype=torch.float16)
    sin = torch.randn(32, device="cuda", dtype=torch.float16)
    kc = torch.zeros(2, 64, 64, device="cuda", dtype=torch.float16)
    vc = torch.zeros(2, 64, 64, device="cuda", dtype=torch.float16)
    qo = torch.empty(14, 1, 64, device="cuda", dtype=torch.float16)
    row("qkv_rope_append", lambda: qkv_rope_append(x1, wqkv, cos, sin, kc, vc, qo, 37),
        1152 * 896 * 2 + 64 * 2, flops=2 * 1152 * 896)

    # embedding gather
    we = torch.randn(50000, 896, device="cuda", dtype=torch.float16)
    idx = torch.randint(0, 50000, (8,), device="cuda")
    row("embedding 8x896", lambda: embedding_lookup(idx, we), 8 * 896 * 2)

    # int8 prefill matmul (compute-bound): 64x2048x2048
    w = torch.randn(2048, 2048, device="cuda", dtype=torch.float16) * 0.05
    x = torch.randn(64, 2048, device="cuda", dtype=torch.float16)
    wi8, sc = quantize_int8(w, 128)
    row("int8 matmul 64x2048x2048", lambda: int8_matmul(x, wi8, sc),
        2048 * 2048 + 64 * 2048 * 2, flops=2 * 64 * 2048 * 2048)

    # Triton FA prefill (compute-bound)
    q = torch.randn(1, 8, 256, 64, device="cuda", dtype=torch.float16)
    k = torch.randn(1, 8, 256, 64, device="cuda", dtype=torch.float16)
    v = torch.randn(1, 8, 256, 64, device="cuda", dtype=torch.float16)
    row("fa prefill S=256", lambda: FA.flash_attn_func(q, k, v, causal=True),
        3 * 8 * 256 * 64 * 2, flops=4 * 256 * 256 * 64 * 8)

    print(f"\npeak used: {PEAK_GBS} GB/s (RDNA1 8GB)", flush=True)
    out = Path(__file__).resolve().parents[1] / "bench" / "results" / "roofline.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"rows": ROWS}, indent=1), encoding="utf-8")
    print(f"report: {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
