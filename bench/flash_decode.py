"""Validate and benchmark flash-decoding (split-KV) attention.

Decode shapes: M=1..8 query tokens, KV length 1k..8k, B1 H8 D64 fp16.
Reference: float32 scores/softmax (masked when causal; the M query tokens are
the last M positions). Baseline: math SDPA (`SDPBackend.MATH`).

Usage:
  run-zluda.cmd venv\\Scripts\\python.exe bench\\flash_decode.py --quick
  run-zluda.cmd venv\\Scripts\\python.exe bench\\flash_decode.py
"""

import argparse
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zluda_ops.flash_decode import flash_decode_attn, _choose_splits
from bench.common import bench_median, save_report


def reference(q, k, v, causal):
    b, h, m, d = q.shape
    n = k.shape[2]
    scale = 1.0 / math.sqrt(d)
    scores = (q.float() @ k.float().transpose(-1, -2)) * scale
    if causal:
        qpos = torch.arange(n - m, n, device=q.device)[:, None]
        kpos = torch.arange(n, device=q.device)[None, :]
        scores = scores.masked_fill(qpos < kpos, float("-inf"))
    p = torch.softmax(scores, dim=-1)
    return p @ v.float()


def math_sdpa(q, k, v, causal):
    mask = None
    if causal:
        n, m = k.shape[2], q.shape[2]
        qpos = torch.arange(n - m, n, device=q.device)[:, None]
        kpos = torch.arange(n, device=q.device)[None, :]
        mask = (qpos >= kpos)[None, None]
    with sdpa_kernel(SDPBackend.MATH):
        return F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False
        )


def check(m, n, causal, device="cuda"):
    torch.manual_seed(0)
    b, h, d = 1, 8, 64
    q = torch.randn(b, h, m, d, device=device, dtype=torch.float16)
    k = torch.randn(b, h, n, d, device=device, dtype=torch.float16)
    v = torch.randn(b, h, n, d, device=device, dtype=torch.float16)
    out = flash_decode_attn(q, k, v, causal=causal)
    torch.cuda.synchronize()
    ref = reference(q, k, v, causal)
    err = (out.float() - ref).abs().max().item()
    rel = err / ref.abs().max().item()
    splits = _choose_splits(n, b * h)
    print(f"check m={m} n={n} causal={int(causal)}: max abs {err:.4f} "
          f"(rel {rel:.2e}, splits {splits})", flush=True)
    return {"m": m, "n": n, "causal": causal, "max_abs": err, "relative": rel,
            "splits": splits}


def bench(m, n, causal, device="cuda"):
    torch.manual_seed(0)
    b, h, d = 1, 8, 64
    q = torch.randn(b, h, m, d, device=device, dtype=torch.float16)
    k = torch.randn(b, h, n, d, device=device, dtype=torch.float16)
    v = torch.randn(b, h, n, d, device=device, dtype=torch.float16)
    fast_ms = bench_median(lambda: flash_decode_attn(q, k, v, causal=causal),
                           rounds=3, iters=5)
    math_ms = bench_median(lambda: math_sdpa(q, k, v, causal=causal),
                           rounds=3, iters=5)
    # bytes: read K and V once, write O
    gb = (2 * h * n * d * 2) / 1e9
    gbs = gb / (fast_ms / 1000.0)
    print(f"bench m={m:2d} n={n:4d} causal={int(causal)}: flash-decoding "
          f"{fast_ms:7.3f} ms ({gbs:6.1f} GB/s) vs math {math_ms:7.3f} ms "
          f"= {math_ms / fast_ms:4.2f}x", flush=True)
    return {"m": m, "n": n, "causal": causal, "fast_ms": fast_ms,
            "math_ms": math_ms, "speedup": math_ms / fast_ms, "gbs": gbs}



def auto_check(m=1, n=4096, device="cuda"):
    """Verify that zluda_ops.auto routes decode SDPA through flash-decoding."""
    import torch.nn.functional as F
    import zluda_ops.auto as auto_mod
    from zluda_ops import scaled_dot_product_attention

    torch.manual_seed(0)
    q = torch.randn(1, 8, m, 64, device=device, dtype=torch.float16)
    k = torch.randn(1, 8, n, 64, device=device, dtype=torch.float16)
    v = torch.randn(1, 8, n, 64, device=device, dtype=torch.float16)
    ref = reference(q, k, v, False)
    auto_mod.install()
    try:
        # no_grad: the auto patch only speeds up inference
        with torch.no_grad():
            out = F.scaled_dot_product_attention(q, k, v)
            direct = flash_decode_attn(q, k, v, causal=False)
        err = (out.float() - ref).abs().max().item()
        same = torch.equal(out, direct)
        math_ms = bench_median(lambda: math_sdpa(q, k, v, False), rounds=2, iters=3)
        auto_ms = bench_median(
            lambda: F.scaled_dot_product_attention(q, k, v), rounds=2, iters=3)
    finally:
        auto_mod.uninstall()
    print(f"auto dispatch m={m} n={n}: err {err:.4f}, matches flash_decode {same}, "
          f"{auto_ms:.3f} ms vs math {math_ms:.3f} ms = {math_ms / auto_ms:.2f}x",
          flush=True)
    return {"m": m, "n": n, "err": err, "matches_direct": same,
            "auto_ms": auto_ms, "math_ms": math_ms, "speedup": math_ms / auto_ms}

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

    checks = [
        check(1, 1024, False),
        check(1, 1024, True),
        check(8, 1024, True),
    ]
    benches = [bench(1, 1024, False), bench(1, 2048, False),
               bench(1, 4096, False), bench(8, 4096, True)]
    if not args.quick:
        checks += [check(1, 4096, False), check(4, 4096, True)]
        benches += [bench(8, 2048, False), bench(4, 4096, False)]
    auto = auto_check() if not args.quick else None
    save_report("flash_decode", {"checks": checks, "benches": benches, "auto": auto})
    worst = max(c["relative"] for c in checks)
    speed4096 = max(b["speedup"] for b in benches if b["n"] >= 4096)
    print(f"worst relative error {worst:.2e}; best 4k speedup {speed4096:.2f}x", flush=True)
    return 0 if worst < 1e-2 else 1


if __name__ == "__main__":
    sys.exit(main())
