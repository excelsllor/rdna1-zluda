"""Cast-to-fp32 convolution workaround for gfx1010 (Winograd).

MIOpen picks a fp32 Winograd solver (`ConvBinWinogradRxSf2x3g1`) for 3x3
convolutions, but fp16 falls back to `GemmFwdRest` (a rocBLAS GEMM). Casting
fp16 inputs/weights to fp32, running the fp32 Winograd conv and casting the
result back (`zluda_ops.conv.conv2d`) is faster and at least as accurate.

Timing uses a warmed GPU and shuffled interleaved rounds (the clocks cannot be
fixed on this card, so a naive sequential loop overestimates later variants).

Usage:
  run-zluda.cmd venv\\Scripts\\python.exe bench\\conv_fp16.py --quick
  run-zluda.cmd venv\\Scripts\\python.exe bench\\conv_fp16.py
"""

import argparse
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zluda_ops import conv as zconv
from bench.common import save_report


def timed_shuffled(fns, warm_seconds=1.5, rounds=5, iters=8):
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < warm_seconds:
        for fn in fns.values():
            fn()
        torch.cuda.synchronize()
    rng = random.Random(7)
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


def compare_shape(shape, device="cuda"):
    b, c, h, w, o, k = shape
    torch.manual_seed(0)
    x16 = torch.randn(b, c, h, w, device=device, dtype=torch.float16)
    w16 = torch.randn(o, c, k, k, device=device, dtype=torch.float16)
    b16 = torch.randn(o, device=device, dtype=torch.float16)
    ref32 = F.conv2d(x16.float(), w16.float(), b16.float(), padding=k // 2)
    out = zconv.conv2d(x16, w16, b16, padding=k // 2)
    fwd_err = (out.float() - ref32).abs().max().item()

    # gradient correctness against the fp32 reference
    xg = x16.clone().requires_grad_(True)
    wg = w16.clone().requires_grad_(True)
    bg = b16.clone().requires_grad_(True)
    zconv.conv2d(xg, wg, bg, padding=k // 2).float().mean().backward()
    xr = x16.float().clone().requires_grad_(True)
    wr = w16.float().clone().requires_grad_(True)
    br = b16.float().clone().requires_grad_(True)
    F.conv2d(xr, wr, br, padding=k // 2).mean().backward()
    grad_err = max(
        (xg.grad.float() - xr.grad).abs().max().item(),
        (wg.grad.float() - wr.grad).abs().max().item(),
        (bg.grad.float() - br.grad).abs().max().item(),
    )

    fwd_ms = timed_shuffled({
        "fp16": lambda: F.conv2d(x16, w16, b16, padding=k // 2),
        "cast": lambda: zconv.conv2d(x16, w16, b16, padding=k // 2),
    })

    # training path: fp16 params vs fp32 master weights (cast each step)
    xg16 = x16.clone().requires_grad_(True)
    w32u = w16.detach().float().requires_grad_(True)
    b32u = b16.detach().float().requires_grad_(True)

    def fb_fp16():
        y = F.conv2d(xg16, w16, b16, padding=k // 2)
        y.float().mean().backward()
        xg16.grad = None

    def fb_cast():
        y = F.conv2d(xg16.float(), w32u, b32u, padding=k // 2).half()
        y.float().mean().backward()
        xg16.grad = w32u.grad = b32u.grad = None

    fb_ms = timed_shuffled({"fp16": fb_fp16, "cast": fb_cast}, rounds=4, iters=5)
    print(f"{shape}: fwd fp16 {fwd_ms['fp16']:.3f} vs cast {fwd_ms['cast']:.3f} ms "
          f"({fwd_ms['fp16'] / fwd_ms['cast']:.2f}x) | fwd+bwd fp16 {fb_ms['fp16']:.3f} "
          f"vs cast {fb_ms['cast']:.3f} ms ({fb_ms['fp16'] / fb_ms['cast']:.2f}x) | "
          f"err vs fp32 {fwd_err:.4f}, grad err {grad_err:.2e}", flush=True)
    return {
        "shape": shape,
        "fwd_fp16_ms": fwd_ms["fp16"], "fwd_cast_ms": fwd_ms["cast"],
        "fwd_speedup": fwd_ms["fp16"] / fwd_ms["cast"],
        "fwd_bwd_fp16_ms": fb_ms["fp16"], "fwd_bwd_cast_ms": fb_ms["cast"],
        "fwd_bwd_speedup": fb_ms["fp16"] / fb_ms["cast"],
        "fwd_err_vs_fp32": fwd_err, "grad_err": grad_err,
    }


def unet_block(batch, ch, hw, device="cuda"):
    torch.manual_seed(0)
    x = torch.randn(batch, ch, hw, hw, device=device, dtype=torch.float16)
    conv_w = torch.randn(ch, ch, 3, 3, device=device, dtype=torch.float16)
    conv_b = torch.randn(ch, device=device, dtype=torch.float16)
    gn_w = torch.ones(ch, device=device, dtype=torch.float16)
    gn_b = torch.zeros(ch, device=device, dtype=torch.float16)

    def run_fp16():
        h = F.silu(F.group_norm(x, 32, gn_w, gn_b))
        return F.conv2d(h, conv_w, conv_b, padding=1) + x

    def run_cast():
        h = F.silu(F.group_norm(x, 32, gn_w, gn_b))
        return zconv.conv2d(h, conv_w, conv_b, padding=1) + x

    ms = timed_shuffled({"fp16": run_fp16, "cast": run_cast})
    diff = (run_fp16().float() - run_cast().float()).abs().max().item()
    print(f"unet block {batch}x{ch}x{hw}x{hw}: fp16 {ms['fp16']:.3f} vs "
          f"cast {ms['cast']:.3f} ms ({ms['fp16'] / ms['cast']:.2f}x), diff {diff:.5f}",
          flush=True)
    return {"batch": batch, "ch": ch, "hw": hw, "fp16_ms": ms["fp16"],
            "cast_ms": ms["cast"], "speedup": ms["fp16"] / ms["cast"],
            "diff": diff}


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

    shapes = [
        (1, 64, 56, 56, 64, 3),
        (8, 64, 56, 56, 64, 3),
        (8, 256, 14, 14, 256, 3),
        (2, 256, 64, 64, 256, 3),
        (1, 320, 64, 64, 320, 3),
    ]
    if args.quick:
        shapes = shapes[:2]
    rows = [compare_shape(s) for s in shapes]
    blocks = [unet_block(1, 256, 32)]
    if not args.quick:
        blocks.append(unet_block(2, 320, 64))
    save_report("conv_fp16", {"rows": rows, "blocks": blocks})
    wins = sum(1 for r in rows if r["fwd_speedup"] > 1.0)
    print(f"cast path faster on {wins}/{len(rows)} shapes (fwd)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
