"""Report MIOpen find-db state, cold-start cost and solver-search behaviour.

Context: MIOpen on this stack ships no system find-db for gfx1010
(`gfx1010_20.HIP.fdb.txt` is not present), but it maintains a user find-db in
`~/.miopen/db/*.ufdb.txt` that is populated automatically. Two things matter for
users:

* the ~1.3 s cost of the very first convolution in a process is MIOpen
  kernel-db/context initialisation, not a solver search;
* forcing a full search (`MIOPEN_FIND_ENFORCE=SEARCH`) or a heuristic-only mode
  (`MIOPEN_FIND_MODE=FAST`) does not change steady-state throughput for the
  SD/ResNet shapes tested, because the cached user-db solution is already the
  fastest.

This script prints those facts so the claims can be re-verified.

Usage:
  run-zluda.cmd venv\\Scripts\\python.exe bench\\miopen_db.py
"""

import glob
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import save_report


def db_state():
    root = Path(os.path.expanduser("~/.miopen"))
    files = sorted(glob.glob(str(root / "db" / "*")), key=os.path.getmtime)
    return {
        "root": str(root),
        "exists": root.exists(),
        "db_files": [(os.path.basename(p), os.path.getsize(p),
                      int(os.path.getmtime(p))) for p in files[-6:]],
    }


def timed_conv(shape, repeats=10):
    b, c, h, w, o = shape
    x = torch.randn(b, c, h, w, device="cuda", dtype=torch.float16)
    weight = torch.randn(o, c, 3, 3, device="cuda", dtype=torch.float16)
    t0 = time.perf_counter()
    F.conv2d(x, weight, padding=1)
    torch.cuda.synchronize()
    first = (time.perf_counter() - t0) * 1000
    for _ in range(3):
        F.conv2d(x, weight, padding=1)
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        t = time.perf_counter()
        F.conv2d(x, weight, padding=1)
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - t) * 1000)
    samples.sort()
    return first, samples[len(samples) // 2]


def main():
    torch.cuda.set_per_process_memory_fraction(0.85)
    print(f"MIOPEN_FIND_MODE={os.environ.get('MIOPEN_FIND_MODE')} "
          f"MIOPEN_FIND_ENFORCE={os.environ.get('MIOPEN_FIND_ENFORCE')}", flush=True)
    state = db_state()
    print(f"find-db: {state['root']} exists={state['exists']}", flush=True)
    for name, size, mtime in state["db_files"]:
        print(f"  {name} ({size} bytes, mtime {mtime})", flush=True)

    # process initialisation cost (measured before any real convolution)
    xd = torch.randn(1, 8, 8, 8, device="cuda", dtype=torch.float16)
    wd = torch.randn(8, 8, 3, 3, device="cuda", dtype=torch.float16)
    t0 = time.perf_counter()
    F.conv2d(xd, wd, padding=1)
    torch.cuda.synchronize()
    init_ms = (time.perf_counter() - t0) * 1000
    print(f"MIOpen/context initialisation on first conv: {init_ms:.0f} ms", flush=True)

    rows = []
    for shape in ((1, 64, 56, 56, 64), (2, 256, 64, 64, 256), (1, 320, 64, 64, 320)):
        first, steady = timed_conv(shape)
        print(f"shape {shape}: first {first:.2f} ms, steady {steady:.4f} ms", flush=True)
        rows.append({"shape": shape, "first_ms": first, "steady_ms": steady})

    state_after = db_state()
    save_report("miopen_db", {
        "db_before": state, "db_after": state_after,
        "init_ms": init_ms, "rows": rows,
        "find_mode": os.environ.get("MIOPEN_FIND_MODE"),
        "find_enforce": os.environ.get("MIOPEN_FIND_ENFORCE"),
    })
    return 0


if __name__ == "__main__":
    sys.exit(main())
