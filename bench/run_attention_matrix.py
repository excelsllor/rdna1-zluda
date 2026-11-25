"""Run the attention-matrix benchmark configurations sequentially.

Each configuration runs in its own `attention_matrix.py` process with a hard
timeout; results are parsed from the JSON outputs and summarised to
bench/results/attention_matrix_summary.json. Designed to run as one background
job (no nested process trees on this machine).

Usage:
  venv\\Scripts\\python.exe bench\\run_attention_matrix.py --dtypes fp16 --prefill 1024 --decode 4
  venv\\Scripts\\python.exe bench\\run_attention_matrix.py --dtypes fp16,fp32,bf16 --acc
"""

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "bench" / "results"
OUT_SUMMARY = RESULTS / "attention_matrix_summary.json"


def run_config(args, tag, timeout_s):
    cmd = [str(REPO / "run-zluda.cmd"), "venv\\Scripts\\python.exe",
           str(REPO / "bench" / "attention_matrix.py"), *args, "--tag", tag]
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True,
                              timeout=timeout_s)
        out = (proc.stdout or "") + (proc.stderr or "")
        m = re.search(r"\{.*\n\}", out, re.S)
        payload = json.loads(m.group(0)) if m else {"error": out[-500:]}
    except subprocess.TimeoutExpired:
        payload = {"error": f"timeout after {timeout_s}s"}
    except Exception as exc:  # noqa: BLE001
        payload = {"error": f"{type(exc).__name__}: {exc}"}
    payload["tag"] = tag
    payload["wall_s"] = round(time.perf_counter() - t0, 1)
    print(f"[runner] {tag}: prefill={payload.get('prefill_tok_s')} "
          f"decode={payload.get('decode_tok_s')} err={payload.get('error')} "
          f"wall={payload['wall_s']}s", flush=True)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtypes", default="fp16")
    parser.add_argument("--acc", action="store_true", help="also run fp16+accum")
    parser.add_argument("--attn", default="math,sdpa_fast,triton_fa,flash_attn_shim,xformers_shim,flex")
    parser.add_argument("--prefill", type=int, default=1024)
    parser.add_argument("--decode", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    old_rows = []
    if OUT_SUMMARY.exists():
        try:
            old_rows = json.loads(OUT_SUMMARY.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            old_rows = []
    rows = [r for r in old_rows if r.get("tag") not in {None}]

    dtypes = [d for d in args.dtypes.split(",") if d]
    attns = [a for a in args.attn.split(",") if a]
    for dtype in dtypes:
        for attn in attns:
            configs = [{"dtype": dtype, "attn": attn, "acc": 0}]
            if args.acc and dtype == "fp16":
                configs.append({"dtype": dtype, "attn": attn, "acc": 1})
            for cfg in configs:
                tag = f"{dtype}_{attn}_acc{cfg['acc']}"
                argv = ["--dtype", dtype, "--attn", attn,
                        "--prefill", str(args.prefill), "--decode", str(args.decode)]
                if cfg["acc"]:
                    argv += ["--acc", "1"]
                if args.compile:
                    argv += ["--compile"]
                rows = [r for r in rows if r.get("tag") != tag]
                rows.append(run_config(argv, tag, args.timeout))
                # persist after every config so a later crash keeps progress
                OUT_SUMMARY.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"[runner] done, {len(rows)} configs -> {OUT_SUMMARY}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
