import os
import shutil
from pathlib import Path

repo = Path(__file__).resolve().parents[1]
line = (
    'import os,sys; e=os.environ.get("ZLUDA_AUTO_OPS")=="1"; '
    f'sys.path.insert(0, r"{repo}") if e and r"{repo}" not in sys.path else None; '
    '__import__("zluda_ops.auto") if e else None\n'
)
for name in ("venv",):
    site = repo / name / "Lib" / "site-packages"
    if site.exists():
        target = site / "zluda_auto.pth"
        target.write_text(line, encoding="utf-8")
        print(f"installed {target}")

stale = repo / "venv-comfy" / "Lib" / "site-packages" / "zluda_auto.pth"
if stale.exists():
    stale.unlink()
    print(f"removed {stale}")

comfy = Path(os.environ.get("COMFYUI_ROOT", r"D:\comfyui\Data\Packages\ComfyUI"))
src = repo / "integrations" / "comfyui" / "zluda_ops_auto"
if comfy.exists():
    dest = comfy / "custom_nodes" / "zluda_ops_auto"
    shutil.copytree(src, dest, dirs_exist_ok=True)
    print(f"installed {dest}")
