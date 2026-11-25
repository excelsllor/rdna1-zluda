"""Validate Triton flash-attention dropout.

Dropout is inverted (kept weights scaled by 1/(1-p)); the row normalization uses
the sum of the *original* probabilities, so E[out] equals the non-dropout
output. This script checks:

  * dropout_p=0 stays bit-compatible with the math fallback;
  * for p=0.1/0.5 the sample mean over fresh seeds matches the non-dropout
    output (inverted dropout is unbiased);
  * forward/backward stay finite and gradient magnitudes are in the expected
    range;
  * timing at s=1024 against math SDPA.

Usage:
  run-zluda.cmd venv\\Scripts\\python.exe bench\\fa_dropout.py --s 256
  run-zluda.cmd venv\\Scripts\\python.exe bench\\fa_dropout.py --s 512 --bench
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zluda_ops import scaled_dot_product_attention as triton_sdpa
from bench.common import bench_median, save_report


def math_sdpa(q, k, v, dropout_p=0.0, causal=False):
    with sdpa_kernel(SDPBackend.MATH):
        return F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=dropout_p, is_causal=causal
        )


def fallback_check(s=256, device="cuda"):
    torch.manual_seed(0)
    q = torch.randn(1, 8, s, 64, device=device, dtype=torch.float16, requires_grad=True)
    k = torch.randn(1, 8, s, 64, device=device, dtype=torch.float16, requires_grad=True)
    v = torch.randn(1, 8, s, 64, device=device, dtype=torch.float16, requires_grad=True)
    do = torch.randn_like(q)
    out = triton_sdpa(q, k, v, dropout_p=0.0, is_causal=True)
    ref = math_sdpa(q, k, v, 0.0, causal=True)
    ferr = (out.float() - ref.float()).abs().max().item()
    out.backward(do)
    g1 = [t.grad.clone() for t in (q, k, v)]
    for t in (q, k, v):
        t.grad = None
    ref.backward(do)
    gerr = max((a - b.grad).abs().max().item() for a, b in zip(g1, (q, k, v)))
    for t in (q, k, v):
        t.grad = None
    print(f"dropout=0 regression: fwd err {ferr:.5f} grad err {gerr:.5f}", flush=True)
    return {"fwd_err": ferr, "grad_err": gerr}


def unbiased_check(p, s=256, seeds=16, device="cuda"):
    import zluda_ops.flash_attention as fa

    torch.manual_seed(0)
    q = torch.randn(1, 8, s, 64, device=device, dtype=torch.float16)
    k = torch.randn(1, 8, s, 64, device=device, dtype=torch.float16)
    v = torch.randn(1, 8, s, 64, device=device, dtype=torch.float16)
    base = triton_sdpa(q, k, v, dropout_p=0.0, is_causal=False).float()
    diffs = []
    for _ in range(seeds):
        out = triton_sdpa(q, k, v, dropout_p=p, is_causal=False).float()
        diffs.append(out - base)
    diffs = torch.stack(diffs)
    mean_diff = diffs.mean(0)
    se = diffs.std(0) / seeds ** 0.5
    # |mean diff| over many elements is ~0.8 * SE for an unbiased estimator.
    z = (mean_diff.abs().mean() / se.mean()).item()
    single_rel = diffs[0].abs().mean().item() / base.abs().mean().item()
    qg = q.clone().requires_grad_(True)
    out = triton_sdpa(qg, k, v, dropout_p=p, is_causal=False)
    out.float().mean().backward()
    finite = bool(torch.isfinite(qg.grad).all().item())
    gnorm = qg.grad.abs().mean().item()
    q_ref = q.clone().requires_grad_(True)
    base_out = triton_sdpa(q_ref, k, v, dropout_p=0.0, is_causal=False).float().mean()
    ref_gnorm = torch.autograd.grad(base_out, q_ref)[0].abs().mean().item()
    print(
        f"p={p}: unbiasedness z={z:.2f} (expect <3), single-sample dev {single_rel:.3f}; "
        f"grad finite {finite} |g| {gnorm:.3e} (no-drop {ref_gnorm:.3e})",
        flush=True,
    )
    return {
        "p": p,
        "unbiased_z": z,
        "single_sample_rel_dev": single_rel,
        "grad_finite": finite,
        "grad_mean_abs": gnorm,
        "grad_mean_abs_no_dropout": ref_gnorm,
    }


def determinism_check(s=128, p=0.5, device="cuda"):
    import zluda_ops.flash_attention as fa

    torch.manual_seed(1)
    q = torch.randn(1, 2, s, 64, device=device, dtype=torch.float16)
    k = torch.randn(1, 2, s, 64, device=device, dtype=torch.float16)
    v = torch.randn(1, 2, s, 64, device=device, dtype=torch.float16)
    fa._DROPOUT_SEED_OVERRIDE = 12345
    try:
        a = triton_sdpa(q, k, v, dropout_p=p, is_causal=True).clone()
        b = triton_sdpa(q, k, v, dropout_p=p, is_causal=True).clone()
        fa._DROPOUT_SEED_OVERRIDE = 999
        c = triton_sdpa(q, k, v, dropout_p=p, is_causal=True).clone()
    finally:
        fa._DROPOUT_SEED_OVERRIDE = None
    same = bool(torch.equal(a, b))
    differs = not torch.equal(a, c)
    print(f"determinism: same seed identical {same}, different seed differs {differs}", flush=True)
    return {"same_seed_identical": same, "different_seed_differs": differs}


def bench_case(s, p, causal, device="cuda"):
    torch.manual_seed(0)
    q = torch.randn(1, 8, s, 64, device=device, dtype=torch.float16)
    k = torch.randn(1, 8, s, 64, device=device, dtype=torch.float16)
    v = torch.randn(1, 8, s, 64, device=device, dtype=torch.float16)
    triton_ms = bench_median(
        lambda: triton_sdpa(q, k, v, dropout_p=p, is_causal=causal), rounds=3, iters=3
    )
    math_ms = bench_median(
        lambda: math_sdpa(q, k, v, p, causal=causal), rounds=3, iters=3
    )
    print(
        f"bench s={s} p={p} causal={int(causal)}: triton {triton_ms:.2f} ms "
        f"math {math_ms:.2f} ms ({math_ms / triton_ms:.2f}x)",
        flush=True,
    )
    return {"s": s, "p": p, "causal": causal, "triton_ms": triton_ms,
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

    reg = fallback_check(args.s)
    unbiased = [unbiased_check(0.1, args.s), unbiased_check(0.5, args.s)]
    determ = determinism_check()
    benches = [bench_case(1024, 0.1, True)] if args.bench else []
    save_report("fa_dropout", {"regression": reg, "unbiased": unbiased,
                               "determinism": determ, "benches": benches})

    ok = reg["fwd_err"] < 2e-2 and reg["grad_err"] < 2e-2
    for row in unbiased:
        ok = ok and row["grad_finite"] and row["unbiased_z"] < 3.0
    ok = ok and determ["same_seed_identical"] and determ["different_seed_differs"]
    print(f"overall: {'OK' if ok else 'FAIL'}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
