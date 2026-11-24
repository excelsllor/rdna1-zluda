import os
import runpy
import sys
from pathlib import Path

import torch

repo = Path(__file__).resolve().parents[1]
script = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(repo))
sys.path.insert(0, str(script.parent))
sys.path.insert(0, str(script.parent.parent))

fraction = float(os.environ.get("ZLUDA_VRAM_FRACTION", "0.85"))
try:
    torch.cuda.set_per_process_memory_fraction(fraction)
    print(f"zluda train: VRAM fraction {fraction}", flush=True)
except Exception as exc:
    print(f"zluda train: VRAM limit unavailable: {exc}", flush=True)

if os.environ.get("ZLUDA_AUTO_OPS") == "1":
    import zluda_ops.auto
else:
    from zluda_ops import mha
    mha.install()

sys.argv = [str(script)] + sys.argv[2:]
runpy.run_path(str(script), run_name="__main__")
