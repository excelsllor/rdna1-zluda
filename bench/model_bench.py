"""Benchmark a Hugging Face causal LM on the ZLUDA/PyTorch/Triton engine.

Measures prefill and decode throughput, peak memory and a training step for a
model on gfx1010, across the configurations that matter for this card:

  * dtype: fp16 / bf16 / fp32
  * fp16 accumulation on/off (cublas HH kernels)
  * attention: math SDPA (ZLUDA cannot use PyTorch flash/mem-efficient on
    gfx1010) vs the Triton flash-attention path from zluda_ops
  * torch.compile on/off
  * weight quantization: none / int8 / int4 (zluda_ops.quant; int4 uses the
    fused dequant kernels, int8 uses a cached dequantized copy)

bitsandbytes is not used: its prebuilt CUDA kernels fail on ZLUDA
(`named symbol not found`).

Usage:
  run-zluda.cmd venv\\Scripts\\python.exe bench\\model_bench.py --quick
  run-zluda.cmd venv\\Scripts\\python.exe bench\\model_bench.py --dtype fp16 --prefill 1024 --decode 32
  run-zluda.cmd venv\\Scripts\\python.exe bench\\model_bench.py --dtype bf16 --fp16acc
  run-zluda.cmd venv\\Scripts\\python.exe bench\\model_bench.py --quant int4 --attn triton
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench.common import bench_median, save_report

DEFAULT_MODEL = str(Path(__file__).resolve().parents[1] / "models" / "Qwen2.5-0.5B-Instruct")


def patch_attention(mode):
    """Return an unpatch callable. mode: 'math' | 'triton' | 'default'."""
    orig = F.scaled_dot_product_attention
    if mode == "math":
        def math_sdpa(*args, **kwargs):
            with sdpa_kernel(SDPBackend.MATH):
                return orig(*args, **kwargs)
        F.scaled_dot_product_attention = math_sdpa
    elif mode == "triton":
        from zluda_ops import flash_attention
        F.scaled_dot_product_attention = flash_attention.scaled_dot_product_attention
    return lambda: setattr(F, "scaled_dot_product_attention", orig)


class QuantLinear(torch.nn.Module):
    """nn.Linear replacement with int4/int8 weight-only storage."""

    def __init__(self, linear, bits=4, group=64):
        super().__init__()
        self.bits = bits
        self.group = group
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        w = linear.weight.detach()
        self.bias = None if linear.bias is None else torch.nn.Parameter(
            linear.bias.detach().clone())
        from zluda_ops import quant as q
        if bits == 4:
            self.packed, self.scales = q.quantize_int4(w, group)
            self._q = q
        elif bits == 8:
            self.int8, self.scales = q.quantize_int8(w, 128)
            self._cached_fp16 = None
            self._q = q
        else:
            raise ValueError("bits must be 4 or 8")

    def forward(self, x):
        shape = x.shape
        x2 = x.reshape(-1, shape[-1])
        if self.bits == 4:
            y = self._q.int4_linear(x2, self.packed, self.scales, self.bias, self.group)
            if y is None:
                y = self._q.int4_matmul(x2, self.packed, self.scales, self.bias, self.group)
        else:
            if self._cached_fp16 is None:
                self._cached_fp16 = self._q.dequantize_int8(
                    self.int8, self.scales,
                    (self.out_features, self.in_features), 128)
            y = F.linear(x2, self._cached_fp16, self.bias)
        return y.reshape(*shape[:-1], self.out_features)


def quantize_model(model, bits, group=64):
    count = 0
    for name, child in list(model.named_children()):
        if isinstance(child, torch.nn.Linear):
            setattr(model, name, QuantLinear(child.to(torch.float16), bits, group))
            count += 1
        else:
            count += quantize_model(child, bits, group)
    return count


def load_model(path, dtype, attn):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    kwargs = dict(local_files_only=True, low_cpu_mem_usage=True)
    if dtype is not None:
        kwargs["torch_dtype"] = dtype
    kwargs["attn_implementation"] = "eager" if attn == "math" else "sdpa"
    model = AutoModelForCausalLM.from_pretrained(path, **kwargs)
    model.eval()
    return tokenizer, model


def measure_prefill(model, input_ids, warmup=2, iters=3):
    def run():
        with torch.no_grad():
            return model(input_ids=input_ids, use_cache=True)
    for _ in range(warmup):
        run()
    torch.cuda.synchronize()
    ms = bench_median(run, rounds=iters, iters=1)
    return ms


def measure_decode(model, input_ids, steps, warmup=1):
    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=True)
    past = out.past_key_values
    next_tok = out.logits[:, -1:].argmax(-1)

    def step():
        nonlocal past, next_tok
        out2 = model(input_ids=next_tok, past_key_values=past, use_cache=True)
        past = out2.past_key_values
        next_tok = out2.logits[:, -1:].argmax(-1)

    with torch.no_grad():
        for _ in range(warmup):
            step()
    torch.cuda.synchronize()
    t = time.perf_counter()
    with torch.no_grad():
        for _ in range(steps):
            step()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / steps * 1000


def measure_train_step(model, input_ids, warmup=1, iters=2):
    model.train()
    labels = input_ids.clone()

    def step():
        out = model(input_ids=input_ids, labels=labels)
        out.loss.backward()
        model.zero_grad(set_to_none=True)
        return out.loss

    step()
    torch.cuda.synchronize()
    ms = bench_median(step, rounds=iters, iters=1)
    model.eval()
    return ms


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--attn", default="math", choices=["math", "triton", "default"])
    parser.add_argument("--quant", default="none", choices=["none", "int8", "int4"])
    parser.add_argument("--group", type=int, default=64)
    parser.add_argument("--fp16acc", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--prefill", type=int, default=512)
    parser.add_argument("--decode", type=int, default=16)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--exact-seq", action="store_true",
                        help="use random token ids of exactly --prefill length (for FA tests)")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--vram-fraction", type=float, default=0.85)
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    if args.quick:
        args.prefill = min(args.prefill, 128)
        args.decode = min(args.decode, 4)
    if not torch.cuda.is_available():
        print("CUDA/ZLUDA device is not available")
        return 1
    torch.cuda.set_per_process_memory_fraction(args.vram_fraction)
    torch.cuda.reset_peak_memory_stats()
    dtypes = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    print(f"VRAM fraction {args.vram_fraction} | dtype={args.dtype} attn={args.attn} "
          f"quant={args.quant} fp16acc={args.fp16acc} compile={args.compile}", flush=True)

    unpatch = patch_attention(args.attn)
    try:
        if args.fp16acc:
            torch._C._set_cublas_allow_fp16_accumulation(True)
        tokenizer, model = load_model(args.model, dtypes[args.dtype], args.attn)
        model = model.cuda()
        load_mb = torch.cuda.max_memory_allocated() / 1024 ** 2
        if args.quant != "none":
            n = quantize_model(model, 4 if args.quant == "int4" else 8, args.group)
            print(f"quantized {n} linear layers to {args.quant}", flush=True)
            quant_mb = torch.cuda.max_memory_allocated() / 1024 ** 2
        else:
            quant_mb = None
        if args.compile:
            t0 = time.perf_counter()
            model = torch.compile(model, dynamic=False)
            compile_s = time.perf_counter() - t0
            print(f"torch.compile setup {compile_s:.1f} s", flush=True)
        else:
            compile_s = None

        text = "The quick brown fox jumps over the lazy dog. " * max(1, args.prefill // 10)
        if args.exact_seq:
            vocab = model.config.vocab_size
            torch.manual_seed(0)
            ids = torch.randint(0, vocab, (1, args.prefill), device="cuda")
        else:
            ids = tokenizer(text, return_tensors="pt", truncation=True,
                            max_length=args.prefill).input_ids.cuda()
        seq = ids.shape[1]
        prefill_ms = measure_prefill(model, ids)
        decode_ms = measure_decode(model, ids, args.decode)
        peak_mb = torch.cuda.max_memory_allocated() / 1024 ** 2
        train_ms = measure_train_step(model, ids) if args.train else None
        result = {
            "dtype": args.dtype, "attn": args.attn, "quant": args.quant,
            "fp16acc": args.fp16acc, "compile": args.compile, "tag": args.tag,
            "seq_len": seq, "load_mb": load_mb, "quant_mb": quant_mb,
            "peak_mb": peak_mb, "compile_s": compile_s,
            "prefill_ms": prefill_ms, "prefill_tok_s": seq / prefill_ms * 1000,
            "decode_ms_per_tok": decode_ms,
            "decode_tok_s": 1000 / decode_ms,
            "train_step_ms": train_ms,
        }
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                          for k, v in result.items()}, indent=2), flush=True)
        save_report("model_bench", result)
    finally:
        unpatch()
    return 0


if __name__ == "__main__":
    sys.exit(main())
