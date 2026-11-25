"""Attention-stack matrix benchmark for the ZLUDA/PyTorch engine.

One configuration per process (clean state):
  --dtype {fp16,fp32,bf16}
  --acc 0|1                 fp16 accumulation (cublas HH) - fp16 only
  --attn {math,sdpa_fast,triton_fa,flex,flash_attn_shim,xformers_shim}
  --compile                  torch.compile (mode --compile-mode)
  --prefill N --decode N

Measures prefill/decode throughput, peak VRAM and the correctness of the
configured attention path (logits vs a math-SDPA reference computed with the
same weights/dtype). Unsupported combinations are reported instead of crashing.

Run under tools/gpu_watchdog.ps1 for safety on this machine, e.g.:

  venv\\Scripts\\python.exe bench\\attention_matrix.py --dtype fp16 --attn triton_fa --prefill 2048 --decode 8
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench.common import bench_median, save_report  # noqa: E402

DEFAULT_MODEL = str(Path(__file__).resolve().parents[1] / "models" / "Qwen2.5-0.5B-Instruct")
_ORIG_SDPA_REF = F.scaled_dot_product_attention


def make_attn_patch(mode, model_cfg):
    """Return (patch_fn, unpatch_fn) that installs the attention backend."""
    orig = F.scaled_dot_product_attention
    state = {"calls": 0, "errors": 0, "last_error": None}

    def _orig_sdpa_with(q, k, v, attn_mask, dropout_p, is_causal, scale, enable_gqa):
        from torch.nn.attention import SDPBackend, sdpa_kernel

        with sdpa_kernel(SDPBackend.MATH):
            return _ORIG_SDPA_REF(q, k, v, attn_mask=attn_mask,
                                  dropout_p=dropout_p, is_causal=is_causal,
                                  scale=scale, enable_gqa=enable_gqa)

    if mode == "math":
        from torch.nn.attention import SDPBackend, sdpa_kernel

        def math_sdpa(*args, **kwargs):
            state["calls"] += 1
            with sdpa_kernel(SDPBackend.MATH):
                return orig(*args, **kwargs)

        F.scaled_dot_product_attention = math_sdpa
        return lambda: setattr(F, "scaled_dot_product_attention", orig), state

    if mode == "sdpa_fast":
        def default_sdpa(*args, **kwargs):
            state["calls"] += 1
            try:
                return orig(*args, **kwargs)  # default backend selection
            except Exception as exc:  # noqa: BLE001
                state["errors"] += 1
                state["last_error"] = f"{type(exc).__name__}: {exc}"
                from torch.nn.attention import SDPBackend, sdpa_kernel

                with sdpa_kernel(SDPBackend.MATH):
                    return orig(*args, **kwargs)

        F.scaled_dot_product_attention = default_sdpa
        return lambda: setattr(F, "scaled_dot_product_attention", orig), state

    if mode == "triton_fa":
        import zluda_ops.auto as auto_ops

        auto_ops.install()
        return auto_ops.uninstall, state

    if mode == "flash_attn_shim":
        import flash_attn

        def shim_sdpa(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False,
                      scale=None, enable_gqa=False):
            state["calls"] += 1
            try:
                qq = q.transpose(1, 2).contiguous()
                kk = k.transpose(1, 2).contiguous()
                vv = v.transpose(1, 2).contiguous()
                out = flash_attn.flash_attn_func(qq, kk, vv, dropout_p=dropout_p,
                                                 softmax_scale=scale, causal=is_causal)
                return out.transpose(1, 2).contiguous()
            except Exception as exc:  # noqa: BLE001
                state["errors"] += 1
                state["last_error"] = f"{type(exc).__name__}: {exc}"
                from torch.nn.attention import SDPBackend, sdpa_kernel

                with sdpa_kernel(SDPBackend.MATH):
                    return _orig_sdpa_with(q, k, v, attn_mask, dropout_p,
                                           is_causal, scale, enable_gqa)


        F.scaled_dot_product_attention = shim_sdpa
        return lambda: setattr(F, "scaled_dot_product_attention", orig), state

    if mode == "xformers_shim":
        import xformers.ops as xops

        def xf_sdpa(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False,
                    scale=None, enable_gqa=False):
            state["calls"] += 1
            try:
                qq = q.transpose(1, 2).contiguous()
                kk = k.transpose(1, 2).contiguous()
                vv = v.transpose(1, 2).contiguous()
                bias = xops.LowerTriangularMask() if is_causal else None
                out = xops.memory_efficient_attention(qq, kk, vv, attn_bias=bias,
                                                      p=dropout_p, scale=scale)
                return out.transpose(1, 2).contiguous()
            except Exception as exc:  # noqa: BLE001
                state["errors"] += 1
                state["last_error"] = f"{type(exc).__name__}: {exc}"
                from torch.nn.attention import SDPBackend, sdpa_kernel

                with sdpa_kernel(SDPBackend.MATH):
                    return _orig_sdpa_with(q, k, v, attn_mask, dropout_p,
                                           is_causal, scale, enable_gqa)

        F.scaled_dot_product_attention = xf_sdpa
        return lambda: setattr(F, "scaled_dot_product_attention", orig), state

    if mode == "flex":
        from torch.nn.attention.flex_attention import (create_block_mask,
                                                       flex_attention)
        cache = {}

        def get_mask(b, h, s):
            key = (b, h, s)
            if key not in cache:
                def causal(bm, hm, qi, ki):
                    return qi >= ki

                cache[key] = create_block_mask(causal, B=b, H=h, Q_LEN=s, KV_LEN=s,
                                               device=q0.device, _compile=False)
            return cache[key]

        q0 = torch.zeros(1, 1, 1, 1, device="cuda")

        def flex_sdpa(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False,
                      scale=None, enable_gqa=False):
            state["calls"] += 1
            try:
                b, h, s, d = q.shape
                mask = None
                if is_causal:
                    if (b, h, s) not in cache:
                        def causal(bm, hm, qi, ki):
                            return qi >= ki

                        cache[(b, h, s)] = create_block_mask(
                            causal, B=b, H=h, Q_LEN=s, KV_LEN=s, device=q.device,
                            _compile=False)
                    mask = cache[(b, h, s)]
                out = flex_attention(q, k, v, block_mask=mask, scale=scale)
                return out
            except Exception as exc:  # noqa: BLE001
                state["errors"] += 1
                state["last_error"] = f"{type(exc).__name__}: {exc}"
                from torch.nn.attention import SDPBackend, sdpa_kernel

                with sdpa_kernel(SDPBackend.MATH):
                    return _orig_sdpa_with(q, k, v, attn_mask, dropout_p,
                                           is_causal, scale, enable_gqa)

        F.scaled_dot_product_attention = flex_sdpa
        return lambda: setattr(F, "scaled_dot_product_attention", orig), state

    raise ValueError(f"unknown attention mode {mode}")


def load_model(dtype, path):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    base = dict(local_files_only=True, low_cpu_mem_usage=True,
                attn_implementation="sdpa")
    try:
        model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype, **base)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype, **base)
    model = model.eval()
    return tokenizer, model


def run_forward(model, ids):
    with torch.no_grad():
        return model(input_ids=ids, use_cache=True, logits_to_keep=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "fp32", "bf16"])
    parser.add_argument("--acc", type=int, default=0, choices=[0, 1])
    parser.add_argument("--bf16-emul", action="store_true",
                        help="bf16 weights stored as bf16 but compute in fp32 (emulation)")
    parser.add_argument("--attn", default="math",
                        choices=["math", "sdpa_fast", "triton_fa", "flex",
                                 "flash_attn_shim", "xformers_shim"])
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--compile-mode", default="default")
    parser.add_argument("--prefill", type=int, default=2048)
    parser.add_argument("--decode", type=int, default=8)
    parser.add_argument("--vram-fraction", type=float, default=0.85)
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    torch.manual_seed(0)
    if not torch.cuda.is_available():
        print("CUDA/ZLUDA device is not available")
        return 1
    torch.cuda.set_per_process_memory_fraction(args.vram_fraction)
    torch.cuda.reset_peak_memory_stats()
    dtypes = {"fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16}
    result = {"dtype": args.dtype, "acc": args.acc, "attn": args.attn,
              "bf16_emul": args.bf16_emul,
              "compile": args.compile, "tag": args.tag,
              "prefill": args.prefill, "decode": args.decode}

    if args.acc and args.dtype != "fp16":
        result["error"] = "fp16-accum applies to fp16 only"
        save_report("attention_matrix", result)
        print(json.dumps(result))
        return 0
    if args.acc:
        torch._C._set_cublas_allow_fp16_accumulation(True)

    unpatch = None
    def log(msg):
        print(f"[attn-matrix] {msg}", flush=True)
    try:
        log("loading model")
        tokenizer, model = load_model(dtypes[args.dtype], args.model)
        unpatch, state = make_attn_patch(args.attn, model.config)
        if args.bf16_emul:
            # bf16 storage, fp32 compute: cast every parameter to fp32 after the
            # bf16 round-trip (weights keep bf16 precision, math runs in fp32)
            for prm in model.parameters():
                prm.data = prm.data.float()
            log("bf16-emul: parameters cast to fp32")
        log("attention patched, moving model to GPU")
        model = model.cuda()
        if args.compile:
            model = torch.compile(model, dynamic=False, mode=args.compile_mode)
        ids = torch.randint(0, model.config.vocab_size, (1, args.prefill),
                            device="cuda")
        # correctness reference: math SDPA with the same weights
        ref = None
        if args.attn != "math":
            F.scaled_dot_product_attention = _ref_math_orig()
            try:
                ref = run_forward(model, ids).logits.clone()
            finally:
                unpatch()
                unpatch, state = make_attn_patch(args.attn, model.config)
        log("warmup forward")
        run_forward(model, ids)
        torch.cuda.synchronize()
        log("measuring prefill")
        prefill_ms = bench_median(lambda: run_forward(model, ids), rounds=2, iters=1)
        with torch.no_grad():
            out = model(input_ids=ids, use_cache=True, logits_to_keep=1)
            past = out.past_key_values
            tok = out.logits[:, -1:].argmax(-1)
            for _ in range(args.decode):
                o2 = model(input_ids=tok, past_key_values=past, use_cache=True,
                           logits_to_keep=1)
                past = o2.past_key_values
                tok = o2.logits[:, -1:].argmax(-1)
        log("measuring decode")
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(args.decode):
                o2 = model(input_ids=tok, past_key_values=past, use_cache=True,
                           logits_to_keep=1)
                past = o2.past_key_values
                tok = o2.logits[:, -1:].argmax(-1)
        torch.cuda.synchronize()
        decode_ms = (time.perf_counter() - t0) / max(1, args.decode) * 1000
        result.update({
            "prefill_ms": prefill_ms, "prefill_tok_s": args.prefill / prefill_ms * 1000,
            "decode_ms_per_tok": decode_ms, "decode_tok_s": 1000 / decode_ms,
            "peak_mb": torch.cuda.max_memory_allocated() / 1024**2,
            "attn_calls": state.get("calls"), "attn_errors": state.get("errors"),
            "attn_last_error": state.get("last_error"),
        })
        if ref is not None:
            try:
                out2 = run_forward(model, ids).logits
                result["logits_max_abs_diff"] = float((out2 - ref).abs().max())
            except Exception as exc:  # noqa: BLE001
                result["logits_error"] = f"{type(exc).__name__}: {exc}"
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
    finally:
        if unpatch:
            try:
                unpatch()
            except Exception:
                pass
    save_report("attention_matrix", result)
    print(json.dumps(result, indent=2))
    return 0


def _ref_math_orig():
    """Math-only SDPA wrapper used for the correctness reference."""
    from torch.nn.attention import SDPBackend, sdpa_kernel

    def math_only(q, k, v, *a, **kw):
        with sdpa_kernel(SDPBackend.MATH):
            return _ORIG_SDPA_REF(q, k, v, *a, **kw)

    return math_only


if __name__ == "__main__":
    sys.exit(main())
