"""Slim smoke for the ecosystem shims (flash_attn, xformers, bitsandbytes).

Run under run-zluda.cmd. Prints one OK/FAIL line per check.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

RESULTS = []


def check(name, fn):
    try:
        fn()
        torch.cuda.synchronize()
        RESULTS.append((name, "OK"))
        print(f"OK   {name}", flush=True)
    except Exception as exc:  # noqa: BLE001
        RESULTS.append((name, None))
        print(f"FAIL {name} {type(exc).__name__}: {str(exc)[:120]}", flush=True)


def main():
    if not torch.cuda.is_available():
        print("CUDA/ZLUDA device is not available")
        return 1
    torch.cuda.set_per_process_memory_fraction(0.7)

    import flash_attn
    import xformers.ops as xops
    import bitsandbytes as bnb

    def fa_prefill():
        q = torch.randn(1, 8, 256, 64, device="cuda", dtype=torch.float16)
        k = torch.randn(1, 8, 256, 64, device="cuda", dtype=torch.float16)
        v = torch.randn(1, 8, 256, 64, device="cuda", dtype=torch.float16)
        out = flash_attn.flash_attn_func(q.transpose(1, 2), k.transpose(1, 2),
                                         v.transpose(1, 2), causal=True)
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            ref = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, is_causal=True)
        err = float((out.transpose(1, 2) - ref).abs().max())
        assert err < 0.05, err

    def fa_decode():
        q = torch.randn(1, 14, 1, 64, device="cuda", dtype=torch.float16)
        k = torch.randn(1, 2, 1024, 64, device="cuda", dtype=torch.float16)
        v = torch.randn(1, 2, 1024, 64, device="cuda", dtype=torch.float16)
        out = flash_attn.flash_attn_func(q.transpose(1, 2), k.transpose(1, 2),
                                         v.transpose(1, 2), causal=False)
        assert bool(torch.isfinite(out).all())

    def xf_decode():
        q = torch.randn(1, 14, 1, 64, device="cuda", dtype=torch.float16)
        k = torch.randn(1, 2, 1024, 64, device="cuda", dtype=torch.float16)
        v = torch.randn(1, 2, 1024, 64, device="cuda", dtype=torch.float16)
        out = xops.memory_efficient_attention(q, k, v)
        assert bool(torch.isfinite(out).all())

    def bnb_opt():
        p = torch.nn.Parameter(torch.randn(1024, device="cuda"))
        opt = bnb.optim.AdamW8bit([p], lr=1e-3)
        p.grad = torch.randn_like(p) * 0.01
        opt.step()
        assert bool(torch.isfinite(p).all())

    check("flash_attn_prefill", fa_prefill)
    check("flash_attn_decode_gqa", fa_decode)
    check("xformers_decode_gqa", xf_decode)
    check("bitsandbytes_adamw8bit", bnb_opt)
    ok = sum(1 for _, s in RESULTS if s == "OK")
    print(f"SUMMARY {ok}/{len(RESULTS)} shim checks passed", flush=True)
    return 0 if ok == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
