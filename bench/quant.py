"""Validate int4/int8 weight-only quantization and the fused int4 dequant-GEMV.

Checks quantization error and compression, correctness of `int4_linear` against a
dequantize-then-linear reference, and decode-speed against the fp16 GEMV kernel
and torch. Peak bandwidth reference: 448 GB/s.

Usage:
  run-zluda.cmd venv\\Scripts\\python.exe bench\\quant.py --quick
  run-zluda.cmd venv\\Scripts\\python.exe bench\\quant.py --n 4096 --k 4096
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zluda_ops import gemv as zgemv
from zluda_ops import quant
from bench.common import bench_median, save_report

PEAK_BW_GBS = 448.0


def quant_error(n=1024, k=1024):
    torch.manual_seed(0)
    w = torch.randn(n, k, device="cuda", dtype=torch.float16) * 0.02
    rows = []
    for bits, group in ((4, 32), (4, 64), (4, 128), (8, 128)):
        if bits == 4:
            packed, scales = quant.quantize_int4(w, group)
            deq = quant.dequantize_int4(packed, scales, (n, k), group)
        else:
            packed, scales = quant.quantize_int8(w, group)
            deq = quant.dequantize_int8(packed, scales, (n, k), group)
        diff = (deq.float() - w.float())
        rel = (diff.norm() / w.float().norm()).item()
        fp16_bytes = w.numel() * 2
        q_bytes = packed.numel() + scales.numel() * 2
        rows.append({"bits": bits, "group": group, "rel_rms": rel,
                     "compression": fp16_bytes / q_bytes,
                     "max_abs": diff.abs().max().item()})
        print(f"int{bits} g{group}: rel RMS {rel:.4%}  compression {fp16_bytes / q_bytes:.2f}x",
              flush=True)
    return rows


def correctness(n=1024, k=1024, group=64):
    torch.manual_seed(0)
    w = torch.randn(n, k, device="cuda", dtype=torch.float16) * 0.02
    bias = (torch.randn(n, device="cuda", dtype=torch.float16) * 0.02)
    packed, scales = quant.quantize_int4(w, group)
    rows = []
    for m in (1, 4, 8, 16):
        x = torch.randn(m, k, device="cuda", dtype=torch.float16) * 0.1
        y = quant.int4_linear(x, packed, scales, bias, group)
        if y is None:
            print(f"m={m}: unsupported", flush=True)
            continue
        ref = quant.linspace_reference(x, packed, scales, bias, group)
        ref32 = F.linear(x.float(), quant.dequantize_int4(packed, scales, (n, k), group).float(),
                         bias.float())
        err = (y.float() - ref.float()).abs().max().item()
        err32 = (y.float() - ref32).abs().max().item()
        rows.append({"m": m, "max_abs_vs_fp16_ref": err, "max_abs_vs_fp32_ref": err32})
        print(f"m={m}: max abs vs fp16-ref {err:.5f}, vs fp32-ref {err32:.5f}", flush=True)
    return rows


def speed(n=4096, k=4096, group=64):
    torch.manual_seed(0)
    w = torch.randn(n, k, device="cuda", dtype=torch.float16) * 0.02
    bias = (torch.randn(n, device="cuda", dtype=torch.float16) * 0.02)
    packed, scales = quant.quantize_int4(w, group)
    w_deq = quant.dequantize_int4(packed, scales, (n, k), group)
    rows = []
    for m in (1, 8):
        x = torch.randn(m, k, device="cuda", dtype=torch.float16) * 0.1
        q_ms = bench_median(lambda: quant.int4_linear(x, packed, scales, bias, group),
                            rounds=3, iters=10)
        fp16_y = zgemv.linear(x, w, bias, accurate=True)
        fp16_ms = bench_median(lambda: zgemv.linear(x, w, bias, accurate=True),
                               rounds=3, iters=10)
        torch_ms = bench_median(lambda: F.linear(x, w_deq, bias), rounds=3, iters=10)
        q_bytes = packed.numel() + scales.numel() * 2 + m * k * 2 + m * n * 2 + n * 2
        q_gbs = q_bytes / (q_ms / 1000.0) / 1e9
        fp16_bytes = n * k * 2 + m * k * 2 + m * n * 2 + n * 2
        fp16_gbs = fp16_bytes / (fp16_ms / 1000.0) / 1e9
        err = (quant.int4_linear(x, packed, scales, bias, group).float()
               - F.linear(x, w_deq, bias).float()).abs().max().item()
        rows.append({"m": m, "int4_ms": q_ms, "int4_gbs": q_gbs,
                     "fp16_ms": fp16_ms, "fp16_gbs": fp16_gbs,
                     "torch_ms": torch_ms, "max_abs": err,
                     "speedup_vs_fp16": fp16_ms / q_ms})
        print(f"m={m}: int4 {q_ms:7.3f} ms ({q_gbs:6.1f} GB/s) vs fp16 GEMV "
              f"{fp16_ms:7.3f} ms ({fp16_gbs:6.1f} GB/s) = {fp16_ms / q_ms:4.2f}x"
              f" | torch(dequant+linear) {torch_ms:7.3f} ms err {err:.4f}", flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--k", type=int, default=4096)
    parser.add_argument("--group", type=int, default=64)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--vram-fraction", type=float, default=0.85)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        print("CUDA/ZLUDA device is not available")
        return 1
    torch.cuda.set_per_process_memory_fraction(args.vram_fraction)
    print(f"VRAM fraction {args.vram_fraction}", flush=True)

    n = 1024 if args.quick else args.n
    k = 1024 if args.quick else args.k
    errs = quant_error(n, k)
    corr = correctness(n, k, args.group)
    spd = speed(n, k, args.group)
    vram = {
        "7b_fp16_gb": round(7e9 * 2 / 1e9, 2),
        "7b_int4_gb": round(7e9 / 2 / 1e9, 2),
        "7b_int8_gb": round(7e9 * 1 / 1e9, 2),
    }
    print(f"7B weights: fp16 {vram['7b_fp16_gb']} GB, int8 {vram['7b_int8_gb']} GB, "
          f"int4 {vram['7b_int4_gb']} GB", flush=True)
    save_report("quant", {"quant_error": errs, "correctness": corr,
                          "speed": spd, "vram_estimate": vram})
    worst = max(c["max_abs_vs_fp32_ref"] for c in corr)
    print(f"worst fused-vs-fp32 error {worst:.5f}", flush=True)
    return 0 if worst < 5e-2 else 1


if __name__ == "__main__":
    sys.exit(main())
