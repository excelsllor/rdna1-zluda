"""Max-fitting-context matrix: ZLUDA PyTorch dtype/quant/attention variants.

For one configuration (dtype x fp16-accum x bf16-emul x quant x attention
backend) it probes increasing context lengths and records the largest context
that fits in VRAM together with prefill/decode throughput there.

- bf16-emul: bf16 weights after round-trip, all maths in fp32.
- quant q8/q4: weights are quantized (groupwise symmetric int8/int4) and then
  dequantized into fp32; the model runs fp32 (i.e. "dequantize to fp32").
- attention: math (forced MATH SDPA) or sdpa_fast (PyTorch default; on ZLUDA the
  fast backends are unavailable and it falls back to math).

Run under run-zluda.cmd:
  venv\\Scripts\\python.exe bench/maxctx_matrix.py --dtype fp16 --attn math
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench.common import save_report  # noqa: E402

DEFAULT_MODEL = str(Path(__file__).resolve().parents[1] / "models" / "Qwen2.5-0.5B-Instruct")
DTYPES = {"fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16}


def _math_attn(q, k, v, *a, **kw):
    from torch.nn.attention import SDPBackend, sdpa_kernel

    with sdpa_kernel(SDPBackend.MATH):
        return _orig_sdpa(q, k, v, *a, **kw)


def _fast_attn(q, k, v, *a, **kw):
    return _orig_sdpa(q, k, v, *a, **kw)


_orig_sdpa = torch.nn.functional.scaled_dot_product_attention


def quantize_weights_fp32(model, bits, group):
    """Quantize Linear weights (int8/int4 groupwise) then dequantize into fp32."""
    import torch.nn as nn

    from zluda_ops import quant as Q

    for module in model.modules():
        if not isinstance(module, nn.Linear):
            continue
        w = module.weight.detach().to(torch.float16)
        if bits == 8:
            packed, scales = Q.quantize_int8(w, 128)
            wdq = Q.dequantize_int8(packed, scales, w.shape, 128)
        elif bits == 4:
            packed, scales = Q.quantize_int4(w, group)
            wdq = Q.dequantize_int4(packed, scales, w.shape, group)
        else:
            raise ValueError(bits)
        module.weight = nn.Parameter(wdq.float(), requires_grad=False)
    return model


def load_model(path, dtype):
    from transformers import AutoModelForCausalLM

    kwargs = dict(local_files_only=True, low_cpu_mem_usage=True,
                  attn_implementation="sdpa")
    try:
        model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype, **kwargs)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype, **kwargs)
    return model.eval()


def measure(model, ctx, decode):
    ids = torch.randint(0, model.config.vocab_size, (1, ctx), device="cuda")
    with torch.no_grad():
        t0 = time.perf_counter()
        out = model(input_ids=ids, use_cache=True, logits_to_keep=1)
        torch.cuda.synchronize()
        prefill_ms = (time.perf_counter() - t0) * 1000
        prefill_tok_s = ctx / prefill_ms * 1000
        cache = out.past_key_values
        tok = out.logits[:, -1:].argmax(-1)
        pos = torch.tensor([ctx], device="cuda")
        t0 = time.perf_counter()
        for _ in range(decode):
            o = model(input_ids=tok, past_key_values=cache, use_cache=True,
                      cache_position=pos, logits_to_keep=1)
            tok = o.logits[:, -1:].argmax(-1)
            pos = pos + 1
        torch.cuda.synchronize()
        decode_ms = (time.perf_counter() - t0) * 1000 / decode
    return {"prefill_tok_s": round(prefill_tok_s, 1),
            "decode_tok_s": round(1000 / decode_ms, 1),
            "decode_ms_per_tok": round(decode_ms, 2)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", default="fp16", choices=list(DTYPES))
    parser.add_argument("--acc", type=int, default=0, choices=[0, 1])
    parser.add_argument("--bf16-emul", action="store_true")
    parser.add_argument("--quant", default="none", choices=["none", "int8", "int4"])
    parser.add_argument("--group", type=int, default=64)
    parser.add_argument("--attn", default="math", choices=["math", "sdpa_fast"])
    parser.add_argument("--contexts", default="2048,4096,8192,12288,16384,24576,32768")
    parser.add_argument("--decode", type=int, default=8)
    parser.add_argument("--vram-fraction", type=float, default=0.85)
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        print("CUDA/ZLUDA device is not available")
        return 1
    torch.cuda.set_per_process_memory_fraction(args.vram_fraction)
    torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(0)

    result = {"dtype": args.dtype, "acc": args.acc, "bf16_emul": args.bf16_emul,
              "quant": args.quant, "attn": args.attn, "tag": args.tag,
              "contexts": [], "max_ok_ctx": None}
    import torch.nn.functional as F

    global _orig_sdpa
    _orig_sdpa = F.scaled_dot_product_attention
    F.scaled_dot_product_attention = _math_attn if args.attn == "math" else _fast_attn
    if args.acc:
        if args.dtype != "fp16":
            result["error"] = "fp16-accum applies to fp16 only"
            print(json.dumps(result))
            return 0
        torch._C._set_cublas_allow_fp16_accumulation(True)

    print("[maxctx] loading model", flush=True)
    model = load_model(DEFAULT_MODEL, DTYPES[args.dtype])
    print("[maxctx] model loaded", flush=True)
    compute_dtype = DTYPES[args.dtype]
    if args.bf16_emul:
        for prm in model.parameters():
            prm.data = prm.data.float()
        compute_dtype = torch.float32
    if args.quant != "none":
        bits = 8 if args.quant == "int8" else 4
        model = quantize_weights_fp32(model, bits, args.group)
        model = model.float()
        compute_dtype = torch.float32
    print("[maxctx] moving to GPU", flush=True)
    model = model.cuda()
    print("[maxctx] on GPU", flush=True)
    result["compute_dtype"] = str(compute_dtype)

    for ctx in [int(c) for c in args.contexts.split(",") if c]:
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            m = measure(model, ctx, args.decode)
            m["ctx"] = ctx
            m["peak_mb"] = round(torch.cuda.max_memory_allocated() / 1024 ** 2, 1)
            result["contexts"].append(m)
            result["max_ok_ctx"] = ctx
            print(f"OK   ctx={ctx:6d} prefill {m['prefill_tok_s']:8.1f} tok/s "
                  f"decode {m['decode_tok_s']:7.1f} tok/s peak {m['peak_mb']:.0f} MB",
                  flush=True)
        except Exception as exc:  # noqa: BLE001
            msg = f"{type(exc).__name__}: {str(exc)[:160]}"
            result["first_fail"] = {"ctx": ctx, "error": msg}
            print(f"FAIL ctx={ctx:6d} {msg}", flush=True)
            torch.cuda.empty_cache()
            break
    save_report("maxctx_matrix", result)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
