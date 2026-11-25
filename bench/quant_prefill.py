"""Prefill comparison: fused int4 dequant-GEMM vs dequant-once + GEMM.

Decode (M<=16) is memory-bound on the int4 weights and the fused GEMV wins.
Prefill (M=32..256) is compute-bound, so the question is whether dequantising
weights on the fly inside a GEMM tile beats dequantising the weight matrix once
(extra 2 bytes/weight traffic) and running a regular fp16 GEMM. This script
answers that, and verifies correctness against a float32 reference.

Usage:
  run-zluda.cmd venv\\Scripts\\python.exe bench\\quant_prefill.py --quick
  run-zluda.cmd venv\\Scripts\\python.exe bench\\quant_prefill.py
"""

import argparse
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zluda_ops import quant
from bench.common import save_report


def timed_shuffled(fns, warm_seconds=1.5, rounds=5, iters=8):
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < warm_seconds:
        for fn in fns.values():
            fn()
        torch.cuda.synchronize()
    rng = random.Random(11)
    samples = {name: [] for name in fns}
    for _ in range(rounds):
        order = list(fns)
        rng.shuffle(order)
        for name in order:
            fn = fns[name]
            for _ in range(2):
                fn()
            torch.cuda.synchronize()
            t = time.perf_counter()
            for _ in range(iters):
                fn()
            torch.cuda.synchronize()
            samples[name].append((time.perf_counter() - t) / iters * 1000)
    return {name: sorted(v)[len(v) // 2] for name, v in samples.items()}


def case(m, n=4096, k=4096, group=64, device="cuda", dtype=torch.float16):
    torch.manual_seed(0)
    w = torch.randn(n, k, device=device, dtype=dtype) * 0.02
    x = torch.randn(m, k, device=device, dtype=dtype) * 0.1
    bias = torch.randn(n, device=device, dtype=dtype) * 0.02
    packed, scales = quant.quantize_int4(w, group)
    w_deq = quant.dequantize_int4(packed, scales, (n, k), group)

    fused_out = quant.int4_matmul(x, packed, scales, bias, group)
    deq_out = quant.int4_prefill_linear(x, packed, scales, bias, group)
    ref = x.float() @ w_deq.float().t() + bias.float()
    fused_err = (fused_out.float() - ref).abs().max().item()
    deq_err = (deq_out.float() - ref).abs().max().item()

    fns = {
        "fused_int4_gemm": lambda: quant.int4_matmul(x, packed, scales, bias, group),
        "dequant_once_gemm": lambda: quant.int4_prefill_linear(x, packed, scales, bias, group),
        "fp16_weights_gemm": lambda: F.linear(x, w, bias),
    }
    ms = timed_shuffled(fns)
    best = min(ms.values())
    print(f"M={m:3d} N=K={n}: fused {ms['fused_int4_gemm']:7.3f} ms | "
          f"dequant-once {ms['dequant_once_gemm']:7.3f} ms | "
          f"fp16-roofline {ms['fp16_weights_gemm']:7.3f} ms | errors "
          f"{fused_err:.4f}/{deq_err:.4f}", flush=True)
    return {
        "m": m, "n": n, "k": k,
        "fused_ms": ms["fused_int4_gemm"],
        "dequant_once_ms": ms["dequant_once_gemm"],
        "fp16_roofline_ms": ms["fp16_weights_gemm"],
        "fused_err": fused_err, "dequant_err": deq_err,
        "fused_vs_dequant": ms["dequant_once_gemm"] / ms["fused_int4_gemm"],
        "fused_vs_fp16": ms["fp16_weights_gemm"] / ms["fused_int4_gemm"],
        "dequant_vs_fp16": ms["fp16_weights_gemm"] / ms["dequant_once_gemm"],
    }


def decode_reference(device="cuda"):
    """Show the decode side for contrast: fused GEMV vs dequant-once at M=1."""
    torch.manual_seed(0)
    n = k = 4096
    group = 64
    w = torch.randn(n, k, device=device, dtype=torch.float16) * 0.02
    x = torch.randn(1, k, device=device, dtype=torch.float16) * 0.1
    bias = torch.randn(n, device=device, dtype=torch.float16) * 0.02
    packed, scales = quant.quantize_int4(w, group)
    fns = {
        "fused_gemv": lambda: quant.int4_linear(x, packed, scales, bias, group),
        "dequant_once": lambda: quant.int4_prefill_linear(x, packed, scales, bias, group),
        "fp16_roofline": lambda: F.linear(x, w, bias),
    }
    ms = timed_shuffled(fns, rounds=4)
    print(f"M=1 (decode): fused GEMV {ms['fused_gemv']:.4f} ms | dequant-once "
          f"{ms['dequant_once']:.3f} ms | fp16 {ms['fp16_roofline']:.3f} ms", flush=True)
    return {k: v for k, v in ms.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--vram-fraction", type=float, default=0.85)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        print("CUDA/ZLUDA device is not available")
        return 1
    torch.cuda.set_per_process_memory_fraction(args.vram_fraction)
    print(f"VRAM fraction {args.vram_fraction}", flush=True)
    if args.quick:
        sizes, nk = [32, 64], 1024
    else:
        sizes, nk = [32, 64, 128, 256], 4096
    rows = [case(m, nk, nk) for m in sizes]
    decode = decode_reference()
    save_report("quant_prefill", {"rows": rows, "decode": decode})
    return 0


if __name__ == "__main__":
    sys.exit(main())
