"""Validate Triton sliding-window attention and varlen dispatch.

The Triton kernel implements the sliding window arithmetically (no dense mask);
the math reference uses an explicit keep-mask.

Usage:
  run-zluda.cmd venv\\Scripts\\python.exe bench\\fa_window.py --s 256
  run-zluda.cmd venv\\Scripts\\python.exe bench\\fa_window.py --s 512 --bench
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zluda_ops import flash_attn_func, flash_attn_varlen_func
from bench.common import bench_median, save_report


def allowed_mask(sq, sk, causal, left, right, device):
    qi = torch.arange(sq, device=device)[:, None]
    ki = torch.arange(sk, device=device)[None, :]
    di = qi - ki
    allowed = torch.ones((sq, sk), dtype=torch.bool, device=device)
    if causal:
        allowed &= di >= 0
    if left is not None:
        allowed &= di <= left
    if right is not None:
        allowed &= di >= -right
    return allowed


def math_ref(q, k, v, allowed, dropout_p=0.0):
    with sdpa_kernel(SDPBackend.MATH):
        return F.scaled_dot_product_attention(
            q, k, v, attn_mask=allowed, dropout_p=dropout_p, is_causal=False
        )


def check_window(s, causal, left, right, device="cuda"):
    torch.manual_seed(0)
    q = torch.randn(1, 8, s, 64, device=device, dtype=torch.float16, requires_grad=True)
    k = torch.randn(1, 8, s, 64, device=device, dtype=torch.float16, requires_grad=True)
    v = torch.randn(1, 8, s, 64, device=device, dtype=torch.float16, requires_grad=True)
    do = torch.randn_like(q)
    out = flash_attn_func(q, k, v, causal=causal, window_size=(left, right))
    torch.cuda.synchronize()
    ref = math_ref(q, k, v, allowed_mask(s, s, causal, left, right, device))
    ferr = (out.float() - ref.float()).abs().max().item()
    out.backward(do)
    g1 = [t.grad.clone() for t in (q, k, v)]
    for t in (q, k, v):
        t.grad = None
    ref.backward(do)
    gerr = max((a - b.grad).abs().max().item() for a, b in zip(g1, (q, k, v)))
    for t in (q, k, v):
        t.grad = None
    print(
        f"window causal={int(causal)} ({left},{right}) s={s}: "
        f"fwd err {ferr:.5f} grad err {gerr:.5f}",
        flush=True,
    )
    return {"s": s, "causal": causal, "left": left, "right": right,
            "fwd_err": ferr, "grad_err": gerr}


def check_varlen(device="cuda"):
    torch.manual_seed(0)
    lens = [128, 256, 384]
    total = sum(lens)
    h, d = 8, 64
    q = torch.randn(total, h, d, device=device, dtype=torch.float16, requires_grad=True)
    k = torch.randn(total, h, d, device=device, dtype=torch.float16, requires_grad=True)
    v = torch.randn(total, h, d, device=device, dtype=torch.float16, requires_grad=True)
    cu = torch.tensor([0] + list(torch.cumsum(torch.tensor(lens), 0).tolist()),
                      device=device, dtype=torch.int32)
    do = torch.randn_like(q)
    left = 64
    out = flash_attn_varlen_func(q, k, v, cu, cu, causal=True,
                                 window_size=(left, 0), max_seqlen_q=max(lens),
                                 max_seqlen_k=max(lens))
    torch.cuda.synchronize()
    refs = []
    starts = [0] + torch.cumsum(torch.tensor(lens), 0).tolist()
    for i, n in enumerate(lens):
        a, b = int(starts[i]), int(starts[i + 1])
        qi = q[a:b].transpose(0, 1).unsqueeze(0)
        ki = k[a:b].transpose(0, 1).unsqueeze(0)
        vi = v[a:b].transpose(0, 1).unsqueeze(0)
        refs.append(math_ref(qi, ki, vi, allowed_mask(n, n, True, left, 0, device))[0])
    ref = torch.cat([r.transpose(0, 1) for r in refs], dim=0)
    ferr = (out.float() - ref.float()).abs().max().item()
    out.backward(do)
    g1 = [t.grad.clone() for t in (q, k, v)]
    for t in (q, k, v):
        t.grad = None
    ref.backward(do)
    gerr = max((a - b.grad).abs().max().item() for a, b in zip(g1, (q, k, v)))
    print(f"varlen window ({left},0) lens={lens}: fwd err {ferr:.5f} grad err {gerr:.5f}",
          flush=True)
    return {"lens": lens, "fwd_err": ferr, "grad_err": gerr}


def bench_case(s, left, causal=True, device="cuda"):
    torch.manual_seed(0)
    q = torch.randn(1, 8, s, 64, device=device, dtype=torch.float16)
    k = torch.randn(1, 8, s, 64, device=device, dtype=torch.float16)
    v = torch.randn(1, 8, s, 64, device=device, dtype=torch.float16)
    triton_ms = bench_median(
        lambda: flash_attn_func(q, k, v, causal=causal, window_size=(left, 0)),
        rounds=3, iters=3,
    )
    mask = allowed_mask(s, s, causal, left, 0, device)
    math_ms = bench_median(lambda: math_ref(q, k, v, mask), rounds=3, iters=3)
    print(f"bench s={s} window=({left},0) causal={int(causal)}: triton {triton_ms:.2f} ms "
          f"math(mask) {math_ms:.2f} ms ({math_ms / triton_ms:.2f}x)", flush=True)
    return {"s": s, "left": left, "causal": causal, "triton_ms": triton_ms,
            "math_ms": math_ms, "speedup": math_ms / triton_ms}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s", type=int, default=256)
    parser.add_argument("--bench", action="store_true")
    parser.add_argument("--vram-fraction", type=float, default=0.85)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        print("CUDA/ZLUDA device is not available")
        return 1
    torch.cuda.set_per_process_memory_fraction(args.vram_fraction)
    print(f"VRAM fraction {args.vram_fraction}", flush=True)

    checks = [
        check_window(args.s, True, 32, 0),
        check_window(args.s, True, 64, 0),
        check_window(args.s, False, 16, 16),
        check_window(args.s, False, 64, 32),
    ]
    varlen = check_varlen()
    benches = []
    if args.bench:
        benches.append(bench_case(1024, 128))
        benches.append(bench_case(1024, 64, causal=False))
    save_report("fa_window", {"checks": checks, "varlen": varlen, "benches": benches})
    worst = max(max(r["fwd_err"], r["grad_err"]) for r in checks + [varlen])
    print(f"worst error {worst:.5f}", flush=True)
    return 0 if worst < 2e-2 else 1


if __name__ == "__main__":
    sys.exit(main())
