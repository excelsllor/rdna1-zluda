"""GPU regression gate for the ZLUDA/RDNA1 stack (audit item P1).

Runs the existing GPU suites, checks every check is OK and that the measured
kernel timings did not regress beyond a tolerance, and stores a baseline JSON so
future runs can gate on it.

Usage (under run-zluda.cmd):
  venv\\Scripts\\python.exe bench/regression_gate.py            # check vs baseline
  venv\\Scripts\\python.exe bench/regression_gate.py --update   # refresh baseline
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "bench" / "results"
BASELINE = RESULTS / "gate_baseline.json"
GATE_REPORT = RESULTS / "regression_gate.json"


def run_py(script, timeout=900):
    t0 = time.time()
    proc = subprocess.run([sys.executable, str(REPO / "bench" / script)],
                          cwd=str(REPO), capture_output=True, text=True,
                          timeout=timeout)
    out = (proc.stdout or "") + (proc.stderr or "")
    return {"script": script, "rc": proc.returncode,
            "wall_s": round(time.time() - t0, 1),
            "summary": [ln.strip() for ln in out.splitlines()
                        if "SUMMARY" in ln], "tail": out[-800:]}


def newest(pattern):
    files = sorted(RESULTS.glob(pattern), key=lambda p: p.stat().st_mtime,
                   reverse=True)
    return files


def collect_metrics():
    """Collect (check_statuses, timing metrics) from the saved JSON reports."""
    checks = {}
    metrics = {}
    phase2 = RESULTS / "phase2_kernels.json"
    if phase2.exists():
        data = json.loads(phase2.read_text(encoding="utf-8"))
        for r in data.get("results", []):
            checks[r["name"]] = r["status"]
            d = r.get("detail") or {}
            if isinstance(d, dict):
                for k, v in d.items():
                    if isinstance(v, (int, float)) and ("us" in k or "ms" in k):
                        metrics[f"{r['name']}:{k}"] = float(v)
    for f in newest("train_suite-*.json")[:1]:
        data = json.loads(f.read_text(encoding="utf-8"))
        for r in data.get("results", []):
            checks[f"train:{r['name']}"] = r["status"]
    return checks, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--update", action="store_true",
                        help="write the current results as the new baseline")
    parser.add_argument("--tolerance", type=float, default=0.60,
                        help="minimum ratio vs baseline (lower is better, 0.6=40%% slower allowed)")
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    runs = [run_py("phase2_kernels.py")]
    runs.append(run_py("shim_smoke.py"))
    try:
        runs.append(run_py("train_suite.py", timeout=1200))
    except Exception as exc:  # noqa: BLE001
        runs.append({"script": "train_suite.py", "rc": 1,
                     "summary": [], "tail": f"{type(exc).__name__}: {exc}"})

    failures = []
    for r in runs:
        print(f"[gate] {r['script']}: rc={r['rc']} {r.get('summary')}", flush=True)
        if r["rc"] != 0:
            failures.append(f"{r['script']} exited {r['rc']}")
    checks, metrics = collect_metrics()
    bad = [k for k, v in checks.items() if v != "OK"]
    if bad:
        failures.append(f"checks not OK: {bad}")

    baseline = {}
    if BASELINE.exists():
        baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    regressions = []
    if not args.update and baseline:
        bmetrics = baseline.get("metrics", {})
        for key, val in metrics.items():
            if key in bmetrics and bmetrics[key] > 0:
                ratio = val / bmetrics[key]
                if ratio > 1.0 / args.tolerance:
                    regressions.append({"metric": key, "base": bmetrics[key],
                                        "now": val, "ratio": round(ratio, 2)})
        if regressions:
            failures.append(f"timing regressions: {regressions}")

    report = {"tag": args.tag, "checks": checks, "metrics": metrics,
              "regressions": regressions, "failures": failures,
              "ok": not failures, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}
    GATE_REPORT.write_text(json.dumps(report, indent=1), encoding="utf-8")
    if args.update:
        BASELINE.write_text(json.dumps({"metrics": metrics,
                                        "timestamp": report["timestamp"]},
                                       indent=1), encoding="utf-8")
        print(f"[gate] baseline updated ({len(metrics)} metrics)", flush=True)
    print(f"[gate] checks={len(checks)} bad={len(bad)} regressions={len(regressions)}",
          flush=True)
    if failures:
        print("REGRESSION GATE: FAIL")
        for f in failures:
            print("  -", f)
        return 1
    print("REGRESSION GATE: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
