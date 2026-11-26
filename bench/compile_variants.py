"""torch.compile compatibility cases (one per process, crash-isolated).

Usage (under run-zluda.cmd):
  venv\\Scripts\\python.exe bench\\compile_variants.py --case prefill_default
Cases: prefill_default, prefill_dynamic, prefill_reduce, decode_static.
Prints a RESULT json line.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MODEL = str(Path(__file__).resolve().parents[1] / "models" / "Qwen2.5-0.5B-Instruct")


def load_model():
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(
        MODEL, local_files_only=True, low_cpu_mem_usage=True,
        dtype=torch.float16, attn_implementation="sdpa").eval().cuda()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-auto", action="store_true")
    parser.add_argument("--case", required=True,
                        choices=["prefill_default", "prefill_dynamic",
                                 "prefill_reduce", "decode_static"])
    args = parser.parse_args()
    if not torch.cuda.is_available():
        print("RESULT " + json.dumps({"case": args.case, "status": "FAIL",
                                      "error": "no device"}))
        return 1
    torch.cuda.set_per_process_memory_fraction(0.75)
    torch.manual_seed(0)
    if not args.no_auto:
        import zluda_ops.auto  # Triton GEMV + FA paths, SDPA guard

    model = load_model()
    row = {"case": args.case}
    if args.case.startswith("prefill"):
        mode = "default"
        dynamic = False
        if args.case == "prefill_dynamic":
            dynamic = True
        elif args.case == "prefill_reduce":
            mode = "reduce-overhead"
        cm = torch.compile(model, mode=mode, dynamic=dynamic)
        ids = torch.randint(0, model.config.vocab_size, (1, 512), device="cuda")
        out = cm(input_ids=ids, use_cache=True, logits_to_keep=1)
        torch.cuda.synchronize()
        for _ in range(2):
            cm(input_ids=ids, use_cache=True, logits_to_keep=1)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(3):
            cm(input_ids=ids, use_cache=True, logits_to_keep=1)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / 3 * 1e3
        row.update({"status": "OK", "prefill_tok_s": 512 / ms * 1000,
                    "finite": bool(torch.isfinite(out.logits).all())})
    else:
        from transformers.cache_utils import StaticCache

        P = 128
        cm = torch.compile(model, mode="default", dynamic=True)
        cache = StaticCache(config=model.config, max_batch_size=1,
                            max_cache_len=P + 16, device="cuda",
                            dtype=torch.float16)
        ids = torch.randint(0, model.config.vocab_size, (1, P), device="cuda")
        pos = torch.arange(P, device="cuda", dtype=torch.long)
        out = cm(input_ids=ids, past_key_values=cache, cache_position=pos,
                 use_cache=True, logits_to_keep=1)
        tok = out.logits[:, -1:].argmax(-1).clone()
        spos = torch.tensor([P], device="cuda", dtype=torch.long)
        for i in range(3):
            o = cm(input_ids=tok, past_key_values=cache,
                   cache_position=spos + i, use_cache=True, logits_to_keep=1)
            tok = o.logits[:, -1:].argmax(-1)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for i in range(4):
            o = cm(input_ids=tok, past_key_values=cache,
                   cache_position=spos + 3 + i, use_cache=True,
                   logits_to_keep=1)
            tok = o.logits[:, -1:].argmax(-1)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / 4 * 1e3
        row.update({"status": "OK", "decode_ms_per_tok": ms,
                    "decode_tok_s": 1000 / ms,
                    "finite": bool(torch.isfinite(o.logits).all())})
    print("RESULT " + json.dumps(row), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
