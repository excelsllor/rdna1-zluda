"""Graph-replayed decode benchmark for ZLUDA (StaticCache + Triton GEMV path).

The decode step of a transformer is a fixed sequence of ~240 small ops. On
ZLUDA the eager loop is dominated by CPU dispatch and slow small GEMV kernels,
so neither plain PyTorch nor llama.cpp-style efficiency is reached. Two things
are needed and both are measured here:

1. `zluda_ops.auto` patches `F.linear` (Triton GEMV for tiny M, including the
   (1, 1, D) decode layout) and `F.scaled_dot_product_attention` (Triton FA /
   flash-decoding), which removes cuBLAS/cuDNN from the decode step;
2. with that path the whole decode step becomes graph-capture safe, so
   `torch.cuda.CUDAGraph` can replay it: one launch instead of ~240.

The graph replays a fixed step (cache position frozen), so this is a
latency-per-step measurement. Correctness is verified against the eager step.

Usage (under run-zluda.cmd):
  venv\\Scripts\\python.exe bench\\graph_decode.py --prefill 1024 --steps 16
  venv\\Scripts\\python.exe bench\\graph_decode.py --prefill 2048 --steps 16 --no-graph
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


def timed(fn, steps):
    fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / steps * 1000


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prefill", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--vram-fraction", type=float, default=0.85)
    parser.add_argument("--no-graph", action="store_true")
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    torch.cuda.set_per_process_memory_fraction(args.vram_fraction)
    result = {"prefill": args.prefill, "steps": args.steps, "tag": args.tag}

    from transformers import AutoModelForCausalLM
    from transformers.cache_utils import StaticCache

    # auto patches F.linear (Triton GEMV) and F.sdpa (Triton FA/flash-decoding)
    import zluda_ops.auto as auto

    result["auto_installed"] = bool(auto.installed())
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
        out = model(input_ids=prompt, past_key_values=cache,
                    cache_position=pos, use_cache=True, logits_to_keep=1)
    tok = out.logits[:, -1:].argmax(-1).clone()
    spos = torch.tensor([args.prefill], device="cuda", dtype=torch.long)

    def step():
        with torch.no_grad():
            return model(input_ids=tok, past_key_values=cache,
                         cache_position=spos, use_cache=True,
                         logits_to_keep=1)

    ref = step().logits.clone()
    result["eager_ms_per_tok"] = timed(step, args.steps)
    result["eager_tok_s"] = 1000 / result["eager_ms_per_tok"]
    result["peak_mb"] = torch.cuda.max_memory_allocated() / 1024**2

    if not args.no_graph:
        try:
            side = torch.cuda.Stream()
            with torch.cuda.stream(side):
                step()  # warmup on the capture stream
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.stream(side):
                with torch.cuda.graph(g):
                    gout = step()
            torch.cuda.synchronize()
            g.replay()  # capture only records; replay to fill buffers
            torch.cuda.synchronize()
            err = float((gout.logits - ref).abs().max())
            result["graph_logits_max_abs_diff"] = err
            result["graph_ms_per_tok"] = timed(g.replay, args.steps)
            result["graph_tok_s"] = 1000 / result["graph_ms_per_tok"]
            result["graph_ok"] = True
        except Exception as exc:  # noqa: BLE001
            result["graph_ok"] = False
            result["graph_error"] = f"{type(exc).__name__}: {str(exc)[:240]}"

    save_report("graph_decode", result)
    print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v)
                      for k, v in result.items()}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
