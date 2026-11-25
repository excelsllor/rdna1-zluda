"""Winograd / MIOpen convolution audit on gfx1010.

The roadmap asks whether MIOpen's Winograd solvers are available and faster
than implicit GEMM for 3x3 stride-1/dilation-1 convolutions (SD/ResNet shapes).
ZLUDA maps cuDNN calls to MIOpen (`zluda_dnn`), so PyTorch conv2d exercises the
MIOpen solver selection. This script measures forward/backward for a shape
matrix, compares NCHW vs channels_last and fp16 vs fp32, checks correctness
against a CPU reference on a small shape, and can capture the MIOpen solver
choice by re-running one convolution with MIOPEN_LOG_LEVEL set.

Usage:
  run-zluda.cmd venv\\Scripts\\python.exe bench\\conv.py --quick
  run-zluda.cmd venv\\Scripts\\python.exe bench\\conv.py
  run-zluda.cmd venv\\Scripts\\python.exe bench\\conv.py --solver-probe
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import bench_median, save_report

DEV = "cuda"


def make(shape, dtype, channels_last, requires_grad=False):
    b, c, h, w, o, k = shape
    x = torch.randn(b, c, h, w, device=DEV, dtype=dtype, requires_grad=requires_grad)
    weight = torch.randn(o, c, k, k, device=DEV, dtype=dtype, requires_grad=requires_grad)
    bias = torch.randn(o, device=DEV, dtype=dtype)
    if channels_last:
        x = x.to(memory_format=torch.channels_last)
    return x, weight, bias


def fwd_ms(shape, dtype, channels_last, benchmark):
    x, weight, bias = make(shape, dtype, channels_last)
    torch.backends.cudnn.benchmark = benchmark
    b, c, h, w, o, k = shape
    p = k // 2
    run = lambda: F.conv2d(x, weight, bias, stride=1, padding=p)
    run()
    torch.cuda.synchronize()
    return bench_median(run, rounds=3, iters=3)


def fwd_bwd_ms(shape, dtype, channels_last):
    x, weight, bias = make(shape, dtype, channels_last, requires_grad=True)
    b, c, h, w, o, k = shape
    p = k // 2

    def run():
        y = F.conv2d(x, weight, bias, stride=1, padding=p)
        loss = y.float().mean()
        loss.backward()
        x.grad = weight.grad = None

    run()
    torch.cuda.synchronize()
    return bench_median(run, rounds=3, iters=2)


def correctness():
    torch.manual_seed(0)
    x = torch.randn(1, 16, 32, 32, dtype=torch.float32)
    w = torch.randn(32, 16, 3, 3, dtype=torch.float32)
    b = torch.randn(32, dtype=torch.float32)
    ref = F.conv2d(x, w, b, stride=1, padding=1)
    out = F.conv2d(x.cuda(), w.cuda(), b.cuda(), stride=1, padding=1).float().cpu()
    err = (out - ref).abs().max().item()
    rel = err / ref.abs().max().item()
    print(f"correctness 3x3 fp32 vs CPU: max abs {err:.3e} (rel {rel:.3e})", flush=True)
    return {"max_abs": err, "relative": rel}


def run_matrix(shapes, dtypes, layouts):
    rows = []
    for shape in shapes:
        for dtype in dtypes:
            for channels_last in layouts:
                row = {"shape": shape, "dtype": str(dtype).split(".")[-1],
                       "channels_last": channels_last}
                try:
                    row["fwd_ms"] = fwd_ms(shape, dtype, channels_last, False)
                except Exception as exc:
                    row["fwd_error"] = f"{type(exc).__name__}: {str(exc)[:120]}"
                    rows.append(row)
                    print(f"{shape} {row['dtype']} cl={channels_last}: FAIL {row['fwd_error']}",
                          flush=True)
                    continue
                try:
                    row["fwd_bwd_ms"] = fwd_bwd_ms(shape, dtype, channels_last)
                except Exception as exc:
                    row["fwd_bwd_error"] = f"{type(exc).__name__}: {str(exc)[:120]}"
                try:
                    row["fwd_benchmark_ms"] = fwd_ms(shape, dtype, channels_last, True)
                except Exception as exc:
                    row["fwd_benchmark_error"] = f"{type(exc).__name__}: {str(exc)[:120]}"
                torch.backends.cudnn.benchmark = False
                b, c, h, w, o, k = shape
                flops = 2.0 * b * o * h * w * c * k * k
                row["fwd_tflops"] = flops / (row["fwd_ms"] / 1000.0) / 1e12
                rows.append(row)
                parts = [f"{shape} {row['dtype']:8s} cl={int(channels_last)}: "
                         f"fwd {row['fwd_ms']:7.2f} ms"]
                if "fwd_bwd_ms" in row:
                    parts.append(f"fwd+bwd {row['fwd_bwd_ms']:7.2f} ms")
                if "fwd_benchmark_ms" in row:
                    parts.append(f"benchmark {row['fwd_benchmark_ms']:7.2f} ms")
                print(" ".join(parts), flush=True)
    return rows


def solver_probe(shape=(1, 64, 56, 56, 64, 3), dtype=torch.float16):
    """Re-run one conv in a child process with MIOpen logging and grep solvers."""
    script = (
        "import torch, torch.nn.functional as F\n"
        f"b,c,h,w,o,k = {shape}\n"
        "x = torch.randn(b,c,h,w,device='cuda',dtype=torch." + str(dtype).split('.')[-1] + ")\n"
        "w_ = torch.randn(o,c,k,k,device='cuda',dtype=x.dtype)\n"
        "y = F.conv2d(x,w_,padding=k//2)\n"
        "torch.cuda.synchronize()\n"
        "print('conv done', tuple(y.shape), flush=True)\n"
    )
    env = dict(os.environ)
    env["MIOPEN_ENABLE_LOGGING"] = "1"
    env["MIOPEN_LOG_LEVEL"] = os.environ.get("MIOPEN_LOG_LEVEL", "6")
    print(f"solver probe: MIOPEN_LOG_LEVEL={env['MIOPEN_LOG_LEVEL']}", flush=True)
    proc = subprocess.run(
        [sys.executable, "-c", script], env=env, capture_output=True, text=True,
        timeout=900,
    )
    text = (proc.stdout or "") + (proc.stderr or "")
    interesting = [
        line.strip() for line in text.splitlines()
        if re.search(r"winograd|implicit|igemm|GemmFwd|FindSolution|FindConvolution|fdb|solver",
                     line, re.IGNORECASE)
    ]
    tail = interesting[-40:]
    print(f"probe exit {proc.returncode}, {len(interesting)} solver-related lines", flush=True)
    for line in tail:
        print("  " + line[:220], flush=True)
    return {"exit": proc.returncode, "lines": len(interesting), "tail": tail}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--solver-probe", action="store_true")
    parser.add_argument("--vram-fraction", type=float, default=0.85)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        print("CUDA/ZLUDA device is not available")
        return 1
    torch.cuda.set_per_process_memory_fraction(args.vram_fraction)
    torch.backends.cudnn.benchmark = False
    print(f"VRAM fraction {args.vram_fraction}", flush=True)
    print(f"cudnn {torch.backends.cudnn.version()}, channels_last supported: "
          f"{torch.backends.cudnn.is_available()}", flush=True)

    corr = correctness()
    shapes = [
        (1, 64, 56, 56, 64, 3),
        (8, 64, 56, 56, 64, 3),
        (1, 128, 28, 28, 128, 3),
        (8, 256, 14, 14, 256, 3),
        (1, 320, 32, 32, 320, 3),
        (2, 256, 64, 64, 256, 3),
        (1, 320, 64, 64, 320, 3),
    ]
    if args.quick:
        shapes = shapes[:3]
    rows = run_matrix(shapes, (torch.float16, torch.float32), (False, True))
    probe = solver_probe() if args.solver_probe else None
    save_report("conv", {"correctness": corr, "rows": rows, "solver_probe": probe})
    return 0


if __name__ == "__main__":
    sys.exit(main())
