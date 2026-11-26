"""One attention path per process (isolation for crash-prone backends).

Called by tools/run_attention_matrix.ps1; prints a single RESULT json line:
  RESULT {"name": ..., "mode": ..., "status": "OK"/"FAIL", ...}

Usage (under run-zluda.cmd):
  venv\\Scripts\\python.exe bench\\attention_variants.py --mode prefill --only math
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PREFILL = {"math", "sdpa_default", "sdpa_flash", "sdpa_efficient", "sdpa_cudnn",
           "triton_fa", "flex", "flash_attn_shim", "xformers_shim"}
DECODE = {"math", "sdpa_default", "triton_flash_decode", "flash_attn_shim",
          "xformers_shim"}


def timeit(fn, n=10):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1e3


def math_sdpa(q, k, v, causal, gqa=False):
    from torch.nn.attention import SDPBackend, sdpa_kernel

    with sdpa_kernel(SDPBackend.MATH):
        return torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=causal, enable_gqa=gqa)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["prefill", "decode"], required=True)
    parser.add_argument("--only", required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        print("RESULT " + json.dumps({"name": args.only, "mode": args.mode,
                                      "status": "FAIL",
                                      "error": "CUDA/ZLUDA unavailable"}))
        return 1
    torch.cuda.set_per_process_memory_fraction(0.7)
    torch.manual_seed(0)
    from torch.nn.attention import SDPBackend, sdpa_kernel

    row = {"name": args.only, "mode": args.mode}
    if args.mode == "prefill":
        q = torch.randn(1, 8, 256, 64, device="cuda", dtype=torch.float16)
        k = torch.randn(1, 8, 256, 64, device="cuda", dtype=torch.float16)
        v = torch.randn(1, 8, 256, 64, device="cuda", dtype=torch.float16)
        ref = math_sdpa(q, k, v, True)
        fns = {
            "math": lambda: math_sdpa(q, k, v, True),
            "sdpa_default": lambda: torch.nn.functional.scaled_dot_product_attention(
                q, k, v, is_causal=True),
            "sdpa_flash": lambda: sdpa_kernel(SDPBackend.FLASH_ATTENTION).__enter__() or
            torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True),
            "sdpa_efficient": lambda: sdpa_kernel(
                SDPBackend.EFFICIENT_ATTENTION).__enter__() or
            torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True),
            "sdpa_cudnn": lambda: sdpa_kernel(SDPBackend.CUDNN_ATTENTION).__enter__() or
            torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True),
            "triton_fa": lambda: __import__(
                "zluda_ops.flash_attention", fromlist=["x"]
            ).scaled_dot_product_attention(q, k, v, is_causal=True),
        }

        def flex():
            from torch.nn.attention.flex_attention import create_block_mask, flex_attention

            def causal(b, h, qi, ki):
                return qi >= ki

            mask = create_block_mask(causal, B=1, H=8, Q_LEN=256, KV_LEN=256,
                                     device=q.device, _compile=False)
            return flex_attention(q, k, v, block_mask=mask)

        def fa_shim():
            import flash_attn

            return flash_attn.flash_attn_func(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                causal=True).transpose(1, 2)

        def xf():
            import xformers.ops as xops

            return xops.memory_efficient_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                attn_bias=xops.LowerTriangularMask()).transpose(1, 2)

        fns.update({"flex": flex, "flash_attn_shim": fa_shim, "xformers_shim": xf})
        target = fns.get(args.only)
        if target is None:
            raise SystemExit(f"unknown prefill variant {args.only}")
        out = target()
        torch.cuda.synchronize()
        row["max_err"] = float((out.float() - ref.float()).abs().max())
        row["ms"] = timeit(target)
    else:
        q = torch.randn(1, 14, 1, 64, device="cuda", dtype=torch.float16)
        k = torch.randn(1, 2, 1024, 64, device="cuda", dtype=torch.float16)
        v = torch.randn(1, 2, 1024, 64, device="cuda", dtype=torch.float16)
        ref = math_sdpa(q, k, v, False, gqa=True)

        def triton_decode():
            from zluda_ops import flash_decode as fd

            return fd.flash_decode_attn(q, k, v, causal=False)

        def fa_shim():
            import flash_attn

            return flash_attn.flash_attn_func(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                causal=False).transpose(1, 2)

        def xf():
            import xformers.ops as xops

            return xops.memory_efficient_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)

        fns = {
            "math": lambda: math_sdpa(q, k, v, False, gqa=True),
            "sdpa_default": lambda: torch.nn.functional.scaled_dot_product_attention(
                q, k, v, is_causal=False, enable_gqa=True),
            "triton_flash_decode": triton_decode,
            "flash_attn_shim": fa_shim,
            "xformers_shim": xf,
        }
        target = fns.get(args.only)
        if target is None:
            raise SystemExit(f"unknown decode variant {args.only}")
        out = target()
        torch.cuda.synchronize()
        row["max_err"] = float((out.float() - ref.float()).abs().max())
        row["ms"] = timeit(target)
    row["status"] = "OK"
    print("RESULT " + json.dumps(row), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
