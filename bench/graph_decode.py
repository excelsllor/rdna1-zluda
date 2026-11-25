"""Decode benchmark: eager vs CUDA-graph replay on ZLUDA.

Measures the per-token decode cost of Qwen2.5-0.5B two ways:

1. eager: the normal `model(input_ids=1 token, past_key_values=...)` loop;
2. graph: the same single-token forward captured with `torch.cuda.CUDAGraph`
   (side stream) and replayed. This shows how much of the decode cost on ZLUDA
   is Python/kernel-launch overhead rather than GPU work.

The graph variant replays the exact captured step (cache position frozen), so it
is a latency-per-step measurement, not a text-generation benchmark. The eager
variant is measured the same way (fixed KV cache, same position) for a fair
comparison, plus a real generation loop number is reported for reference.

A torch.profiler pass counts the kernels launched per decode step, which is the
main explanation for the Vulkan-vs-ZLUDA decode gap.

Usage (under run-zluda.cmd):
  venv\\Scripts\\python.exe bench\\graph_decode.py --prefill 1024 --steps 32
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prefill", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--vram-fraction", type=float, default=0.85)
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    torch.cuda.set_per_process_memory_fraction(args.vram_fraction)
    result = {"prefill": args.prefill, "steps": args.steps, "tag": args.tag}

    from transformers import AutoModelForCausalLM
    from transformers.cache_utils import StaticCache

    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, low_cpu_mem_usage=True,
        dtype=torch.float16, attn_implementation="sdpa").eval().cuda()
    max_len = args.prefill + args.steps + 8
    cache = StaticCache(config=model.config, max_batch_size=1,
                        max_cache_len=max_len, device="cuda", dtype=torch.float16)
    prompt = torch.randint(0, model.config.vocab_size, (1, args.prefill),
                           device="cuda")
    pos = torch.arange(args.prefill, device="cuda", dtype=torch.long)
    with torch.no_grad():
        out = model(input_ids=prompt, past_key_values=cache, cache_position=pos,
                    use_cache=True, logits_to_keep=1)
    next_tok = out.logits[:, -1:].argmax(-1)
    result["prefill_ok"] = True

    # --- eager: fixed single-token step (same position each time) ---
    step_ids = next_tok.clone()
    step_pos = torch.tensor([args.prefill], device="cuda", dtype=torch.long)

    def eager_step():
        with torch.no_grad():
            return model(input_ids=step_ids, past_key_values=cache,
                         cache_position=step_pos, use_cache=True,
                         logits_to_keep=1)

    eager_step()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(args.steps):
        eager_step()
    torch.cuda.synchronize()
    eager_ms = (time.perf_counter() - t0) / args.steps * 1000
    result["eager_ms_per_tok"] = eager_ms
    result["eager_tok_s"] = 1000 / eager_ms

    # --- CUDA graph replay of one decode step ---
    graph_ok = False
    try:
        side = torch.cuda.Stream()
        with torch.cuda.stream(side):
            eager_step()  # warmup on the capture stream
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.stream(side):
            with torch.cuda.graph(g):
                graph_out = model(input_ids=step_ids, past_key_values=cache,
                                  cache_position=step_pos, use_cache=True,
                                  logits_to_keep=1)
        torch.cuda.synchronize()
        g.replay()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(args.steps):
            g.replay()
        torch.cuda.synchronize()
        graph_ms = (time.perf_counter() - t0) / args.steps * 1000
        result["graph_ms_per_tok"] = graph_ms
        result["graph_tok_s"] = 1000 / graph_ms
        result["graph_ok"] = True
        graph_ok = True
    except Exception as exc:  # noqa: BLE001
        result["graph_error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
        result["graph_ok"] = False

    # --- real generation loop (dynamic cache) ---
    try:
        cache2 = StaticCache(config=model.config, max_batch_size=1,
                             max_cache_len=max_len, device="cuda",
                             dtype=torch.float16)
        with torch.no_grad():
            o = model(input_ids=prompt, past_key_values=cache2, cache_position=pos,
                      use_cache=True, logits_to_keep=1)
            tok = o.logits[:, -1:].argmax(-1)
            p = args.prefill
            t0 = time.perf_counter()
            for _ in range(args.steps):
                o = model(input_ids=tok, past_key_values=cache2,
                          cache_position=torch.tensor([p], device="cuda"),
                          use_cache=True, logits_to_keep=1)
                tok = o.logits[:, -1:].argmax(-1)
                p += 1
        torch.cuda.synchronize()
        gen_ms = (time.perf_counter() - t0) / args.steps * 1000
        result["gen_ms_per_tok"] = gen_ms
        result["gen_tok_s"] = 1000 / gen_ms
    except Exception as exc:  # noqa: BLE001
        result["gen_error"] = f"{type(exc).__name__}: {str(exc)[:160]}"

    # --- profiler: kernels launched per eager decode step ---
    try:
        from torch.profiler import ProfilerActivity, profile

        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(4):
                eager_step()
            torch.cuda.synchronize()
        events = [e for e in prof.events()
                  if getattr(e, "device_type", None) is not None]
        kernels = [e for e in events if getattr(e, "cuda_time_total", 0) > 0]
        total_us = sum(e.cuda_time_total for e in kernels)
        result["profiler_kernels_per_step"] = round(len(kernels) / 4, 1)
        result["profiler_kernel_gpu_ms_per_step"] = round(total_us / 4 / 1000, 3)
    except Exception as exc:  # noqa: BLE001
        result["profiler_error"] = f"{type(exc).__name__}: {str(exc)[:160]}"

    result["peak_mb"] = torch.cuda.max_memory_allocated() / 1024**2
    save_report("graph_decode", result)
    print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v)
                      for k, v in result.items()}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
