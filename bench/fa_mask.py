"""Validate Triton flash-attention masks against the math SDPA fallback.

Cases: boolean 2D mask, additive 2D mask, 4D per-head mask, 3D per-batch mask,
causal+mask, and (for the benchmark) masked attention at s=1024.

Usage:
  run-zluda.cmd venv\\Scripts\\python.exe bench\\fa_mask.py            # s=256 checks
  run-zluda.cmd venv\\Scripts\\python.exe bench\\fa_mask.py --s 512 --bench
"""

import argparse
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zluda_ops import scaled_dot_product_attention as triton_sdpa
from bench.common import bench_median, save_report


def math_sdpa(q, k, v, attn_mask=None, causal=False):
    with sdpa_kernel(SDPBackend.MATH):
        return F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=causal
        )


def make_mask(kind, batch, heads, sq, sk, device):
    eye = torch.eye(sq, sk, device=device, dtype=torch.bool)
    if kind == "bool2d":
        return (torch.rand(sq, sk, device=device) > 0.25) | eye
    if kind == "additive2d":
        return torch.randn(sq, sk, device=device, dtype=torch.float32)
    if kind == "bool4d":
        return (torch.rand(batch, heads, sq, sk, device=device) > 0.25) | eye
    if kind == "additive3d":
        return torch.randn(batch, sq, sk, device=device, dtype=torch.float32)
    raise ValueError(kind)


def combined_mask(mask, s, device):
    """math SDPA forbids attn_mask together with is_causal; merge them."""
    causal_allowed = torch.tril(torch.ones(s, s, device=device, dtype=torch.bool))
    if mask.dtype == torch.bool:
        return (mask & causal_allowed.view(1, 1, s, s)).contiguous()
    big_neg = torch.full((s, s), -1.0e9, device=device, dtype=mask.dtype)
    bias = torch.where(causal_allowed, torch.zeros_like(big_neg), big_neg)
    return mask + bias.view(1, 1, s, s)


def check(kind, s=256, causal=False, device="cuda"):
    torch.manual_seed(0)
    b, h, d = 1, 8, 64
    q = torch.randn(b, h, s, d, device=device, dtype=torch.float16, requires_grad=True)
    k = torch.randn(b, h, s, d, device=device, dtype=torch.float16, requires_grad=True)
    v = torch.randn(b, h, s, d, device=device, dtype=torch.float16, requires_grad=True)
    do = torch.randn_like(q)
    mask = make_mask(kind, b, h, s, s, device)

    out = triton_sdpa(q, k, v, attn_mask=mask, is_causal=causal)
    torch.cuda.synchronize()
    ref = math_sdpa(
        q, k, v, attn_mask=combined_mask(mask, s, device) if causal else mask,
        causal=False,
    )
    ferr = (out.float() - ref.float()).abs().max().item()
    out.backward(do, retain_graph=True)
    gq, gk, gv = q.grad.clone(), k.grad.clone(), v.grad.clone()
    q.grad = k.grad = v.grad = None
    ref.backward(do)
    gerr = max(
        (gq - q.grad).abs().max().item(),
        (gk - k.grad).abs().max().item(),
        (gv - v.grad).abs().max().item(),
    )
    q.grad = k.grad = v.grad = None
    print(
        f"{kind:12s} s={s} causal={int(causal)}: fwd err {ferr:.5f} grad err {gerr:.5f}",
        flush=True,
    )
    return {"kind": kind, "s": s, "causal": causal, "fwd_err": ferr, "grad_err": gerr}


def bench_case(s, causal, device="cuda"):
    b, h, d = 1, 8, 64
    torch.manual_seed(0)
    q = torch.randn(b, h, s, d, device=device, dtype=torch.float16)
    k = torch.randn(b, h, s, d, device=device, dtype=torch.float16)
    v = torch.randn(b, h, s, d, device=device, dtype=torch.float16)
    mask = torch.rand(s, s, device=device) > 0.1
    triton_ms = bench_median(
        lambda: triton_sdpa(q, k, v, attn_mask=mask, is_causal=causal), rounds=3, iters=3
    )
    ref_mask = combined_mask(mask, s, q.device) if causal else mask
    math_ms = bench_median(
        lambda: math_sdpa(q, k, v, attn_mask=ref_mask, causal=False), rounds=3, iters=3
    )
    print(
        f"bench s={s} causal={int(causal)}: triton {triton_ms:.2f} ms math {math_ms:.2f} ms "
        f"speedup {math_ms / triton_ms:.2f}x",
        flush=True,
    )
    return {
        "s": s,
        "causal": causal,
        "triton_ms": triton_ms,
        "math_ms": math_ms,
        "speedup": math_ms / triton_ms,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s", type=int, default=256)
    parser.add_argument("--bench", action="store_true")
    parser.add_argument("--vram-fraction", type=float, default=0.85)
    args = parser.parse_args()

    dtype_ok = torch.cuda.is_available()
    if not dtype_ok:
        print("CUDA/ZLUDA device is not available")
        return 1
    torch.cuda.set_per_process_memory_fraction(args.vram_fraction)
    print(f"VRAM fraction {args.vram_fraction}", flush=True)

    rows = [
        check("bool2d", args.s),
        check("additive2d", args.s),
        check("bool4d", args.s),
        check("additive3d", args.s),
        check("bool2d", args.s, causal=True),
        check("additive2d", args.s, causal=True),
    ]
    benches = []
    if args.bench:
        benches.append(bench_case(1024, True))
        benches.append(bench_case(1024, False))
    save_report("fa_mask", {"checks": rows, "benches": benches, "s": args.s})
    worst = max(max(r["fwd_err"], r["grad_err"]) for r in rows)
    print(f"worst error {worst:.5f}", flush=True)
    return 0 if worst < 2e-2 else 1


if __name__ == "__main__":
    sys.exit(main())
