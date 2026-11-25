"""Benchmark an Ollama model (Vulkan backend on gfx1010) over the HTTP API.

Reports prefill (prompt eval) and decode (eval) throughput, load time and the
GPU/CPU processor split chosen by Ollama. Designed to be compared against the
same GGUF run on llama.cpp-CUDA through ZLUDA, and against the PyTorch/ZLUDA
matrix (`bench/model_bench.py`).

The Ollama server is started/restarted by `bench/ollama_config.ps1` when
switching backend options (flash attention, KV cache type).

Usage:
  venv\\Scripts\\python.exe bench\\ollama_bench.py --model qwen05-bench --ctx 2048,8192,32768
"""

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import save_report

HOST = "http://127.0.0.1:11434"


def post(path, payload, timeout=3600):
    req = urllib.request.Request(HOST + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def get(path, timeout=120):
    with urllib.request.urlopen(HOST + path, timeout=timeout) as r:
        return json.loads(r.read().decode())


def ps():
    try:
        return get("/api/ps").get("models", [])
    except Exception as exc:  # noqa: BLE001
        return [{"error": str(exc)}]


def make_prompt(tokens_approx):
    # ~4 characters per token for English text
    sentence = "The quick brown fox jumps over the lazy dog. "
    reps = max(1, tokens_approx * 4 // len(sentence))
    return (sentence * reps)[: tokens_approx * 4]


def bench_ctx(model, ctx_tokens, num_predict, timeout=3600):
    prompt = make_prompt(ctx_tokens)
    t0 = time.time()
    res = post("/api/generate", {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"num_ctx": ctx_tokens, "num_predict": num_predict,
                    "temperature": 0.0},
    }, timeout=timeout)
    wall = time.time() - t0
    ns = 1e9
    pc = res.get("prompt_eval_count", 0)
    pd = res.get("prompt_eval_duration", 0) / ns
    ec = res.get("eval_count", 0)
    ed = res.get("eval_duration", 0) / ns
    ld = res.get("load_duration", 0) / ns
    row = {
        "requested_ctx": ctx_tokens,
        "prompt_tokens": pc,
        "prompt_ms": pd * 1000,
        "prefill_tok_s": pc / pd if pd else None,
        "eval_tokens": ec,
        "decode_ms_per_tok": (ed / ec * 1000) if ec else None,
        "decode_tok_s": (ec / ed) if ed else None,
        "load_s": ld,
        "wall_s": wall,
    }
    print(f"ctx={ctx_tokens:6d} prompt_tokens={pc:6d} prefill={row['prefill_tok_s'] or 0:8.1f} tok/s "
          f"decode={row['decode_tok_s'] or 0:7.2f} tok/s load={ld:5.1f}s", flush=True)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="qwen05-q4")
    parser.add_argument("--ctx", default="2048,8192,16384,32768")
    parser.add_argument("--num-predict", type=int, default=32)
    parser.add_argument("--unload", action="store_true")
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    ctxs = [int(x) for x in args.ctx.split(",") if x]
    print(f"model={args.model} ctx={ctxs} num_predict={args.num_predict}", flush=True)
    models = ps()
    print("ollama ps before:", json.dumps(models)[:400], flush=True)
    rows = []
    for ctx in ctxs:
        rows.append(bench_ctx(args.model, ctx, args.num_predict))
    if args.unload:
        try:
            post("/api/generate", {"model": args.model, "prompt": "", "keep_alive": 0},
                 timeout=60)
        except Exception:
            pass
    ps_after = ps()
    print("ollama ps after:", json.dumps(ps_after)[:600], flush=True)
    save_report("ollama_bench", {
        "model": args.model, "tag": args.tag, "rows": rows,
        "processor": ps_after, "num_predict": args.num_predict,
    })
    return 0


if __name__ == "__main__":
    sys.exit(main())
