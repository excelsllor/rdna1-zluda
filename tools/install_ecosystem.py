"""Install the ZLUDA ecosystem shims (flash_attn, xformers) into the venv.

Adds a `.pth` file to site-packages that puts the repository root and
`integrations/` on sys.path, so `import flash_attn` / `import xformers` resolve
to the shims backed by the Triton kernels (no CUDA flash-attn/xformers build
needed). Reversible: delete the generated `.pth` file.

Usage:
  venv\\Scripts\\python.exe tools\\install_ecosystem.py
  venv\\Scripts\\python.exe tools\\install_ecosystem.py --uninstall
"""

import argparse
import site
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PTH_NAME = "zluda_ecosystem.pth"


def target_dir():
    candidates = [Path(p) for p in site.getsitepackages()]
    for cand in candidates:
        if cand.exists():
            return cand
    return Path(site.getusersitepackages())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uninstall", action="store_true")
    args = parser.parse_args()
    dest = target_dir() / PTH_NAME
    if args.uninstall:
        if dest.exists():
            dest.unlink()
            print(f"removed {dest}")
        else:
            print(f"{dest} does not exist")
        return 0
    lines = [
        f"{REPO}",
        f"{REPO / 'integrations'}",
    ]
    dest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {dest}")
    print("  paths:")
    for line in lines:
        print(f"    {line}")
    print("verify: venv\\Scripts\\python.exe -c \"import flash_attn, xformers; "
          "print(flash_attn.__version__, xformers.__version__)\"")
    return 0


if __name__ == "__main__":
    sys.exit(main())
