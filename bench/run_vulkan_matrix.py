"""Run the Ollama/Vulkan benchmark matrix and aggregate the results.

For each (model, flash-attention, kv-cache-type) the Ollama server is restarted
with `bench/ollama_config.ps1`, then prefill/decode are measured at every
requested context length. Results are written to one JSON report.

Usage:
  venv\\Scripts\\python.exe bench\\run_vulkan_matrix.py --ctx 2048,8192,32768 --num-predict 16
  venv\\Scripts\\python.exe bench\\run_vulkan_matrix.py --models qwen05-q4 --fa 1 --kv q8_0
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "bench"))
from ollama_bench import bench_ctx, ps  # noqa: E402
from common import save_report  # noqa: E402


def restart_ollama(fa, kv):
    proc = subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
         "-File", str(REPO / "bench" / "ollama_config.ps1"),
         "-FlashAttention", str(fa), "-KvCacheType", kv],
        capture_output=True, text=True, timeout=180,
    )
    ok = proc.returncode == 0
    print(f"ollama restart FA={fa} KV={kv}: {'ok' if ok else 'FAILED'} "
          f"{(proc.stdout or '').strip()[-120:]}", flush=True)
    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default="qwen05-q4,qwen05-f16")
    parser.add_argument("--fa", default="1,0")
    parser.add_argument("--kv", default="q8_0")
    parser.add_argument("--ctx", default="2048,8192,32768")
    parser.add_argument("--num-predict", type=int, default=16)
    args = parser.parse_args()
    models = [m for m in args.models.split(",") if m]
    fas = [int(x) for x in args.fa.split(",") if x]
    kvs = [k for k in args.kv.split(",") if k]
    ctxs = [int(x) for x in args.ctx.split(",") if x]
    rows = []
    for kv in kvs:
        for fa in fas:
            if not restart_ollama(fa, kv):
                rows.append({"error": "restart failed", "fa": fa, "kv": kv})
                continue
            for model in models:
                for ctx in ctxs:
                    try:
                        row = bench_ctx(model, ctx, args.num_predict)
                        row.update({"model": model, "fa": fa, "kv": kv})
                    except Exception as exc:  # noqa: BLE001
                        row = {"model": model, "fa": fa, "kv": kv, "ctx": ctx,
                               "error": f"{type(exc).__name__}: {str(exc)[:160]}"}
                    # record the context Ollama actually allocated
                    try:
                        loaded = [m for m in ps() if m.get("name", "").startswith(model)]
                        row["context_length"] = loaded[0].get("context_length") if loaded else None
                        row["size_vram"] = loaded[0].get("size_vram") if loaded else None
                    except Exception:
                        pass
                    rows.append(row)
                    keys = ('model', 'fa', 'kv', 'requested_ctx', 'context_length',
                            'prefill_tok_s', 'decode_tok_s')
                    summary = {k: row.get(k) for k in keys}
                    print("  -> " + json.dumps(summary), flush=True)
    path = save_report("vulkan_matrix", {"rows": rows})
    print(f"report: {path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
