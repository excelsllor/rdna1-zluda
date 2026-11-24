import argparse
import sys
import warnings
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import bench, save_report
from zluda_ops.fp16acc import fp16_accumulation, enabled

DEV = "cuda"


def measure(n, iters=8):
    a = torch.randn(n, n, device=DEV, dtype=torch.float16)
    b = torch.randn(n, n, device=DEV, dtype=torch.float16)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with fp16_accumulation(False, warn=False):
            ref = a @ b
            ms_fp32 = bench(lambda: a @ b, warmup=2, iters=iters)
        with fp16_accumulation(True, warn=False):
            out = a @ b
            ms_fp16 = bench(lambda: a @ b, warmup=2, iters=iters)
    diff = (out.float() - ref.float()).abs()
    return {
        "n": n,
        "fp32_ms": ms_fp32,
        "fp32_tflops": 2 * n**3 / (ms_fp32 / 1000) / 1e12,
        "fp16_ms": ms_fp16,
        "fp16_tflops": 2 * n**3 / (ms_fp16 / 1000) / 1e12,
        "max_abs_vs_fp32acc": diff.max().item(),
        "mean_abs_vs_fp32acc": diff.mean().item(),
        "relative_max": (diff.max() / ref.float().abs().max().clamp_min(1e-6)).item(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(0)
    print("default fp16 accumulation:", enabled())
    print("NOTE: fp16 accumulation is inference-only and numerically unsafe; validate your model.")
    rows = []
    for n in ([1024, 2048] if args.quick else [1024, 2048, 4096]):
        r = measure(n)
        rows.append(r)
        print(
            f"n={n:5d}: fp32acc {r['fp32_ms']:8.2f} ms / {r['fp32_tflops']:5.2f} TF | "
            f"fp16acc {r['fp16_ms']:8.2f} ms / {r['fp16_tflops']:5.2f} TF | "
            f"max diff {r['max_abs_vs_fp32acc']:.4f} rel {r['relative_max']:.4f}"
        )
    print("final fp16 accumulation:", enabled())
    save_report("fp16_ux", {"rows": rows})


if __name__ == "__main__":
    main()
