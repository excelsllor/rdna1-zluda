"""Benchmark the Triton 8-bit AdamW (bitsandbytes substitute) on ZLUDA.

bitsandbytes' CUDA kernels cannot run through ZLUDA on gfx1010, so
`zluda_ops.optim8bit.AdamW8bit` implements the same idea with a Triton kernel:
int8 moment estimates with per-block fp32 absmax scales.

The script compares step time, optimizer-state memory and parameter parity
against torch.optim.AdamW, and runs a short training loop with both.

Usage:
  run-zluda.cmd venv\\Scripts\\python.exe bench\\optim8bit.py --quick
  run-zluda.cmd venv\\Scripts\\python.exe bench\\optim8bit.py
"""

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zluda_ops.optim8bit import AdamW8bit
from bench.common import bench_median, save_report


def step_case(num_params, device="cuda"):
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.randn(num_params, device=device) * 0.1)
    grads = [torch.randn(num_params, device=device) * 0.01 for _ in range(4)]
    p_t = torch.nn.Parameter(p.detach().clone())
    opt_t = torch.optim.AdamW([p_t], lr=1e-3)
    opt_8 = AdamW8bit([p], lr=1e-3)
    idx = [0]

    def torch_step():
        p_t.grad = grads[idx[0] % len(grads)]
        idx[0] += 1
        opt_t.step()
        opt_t.zero_grad()

    def bit_step():
        p.grad = grads[idx[0] % len(grads)]
        idx[0] += 1
        opt_8.step()

    torch_step()
    bit_step()
    tm = bench_median(torch_step, rounds=3, iters=10)
    bm = bench_median(bit_step, rounds=3, iters=10)
    mem8 = opt_8.state_memory_bytes()
    mem32 = opt_8.fp32_state_memory_bytes()
    print(f"step {num_params} fp32 params: torch {tm:.3f} ms | 8-bit {bm:.3f} ms "
          f"({tm / bm:.2f}x) | state mem {mem8 / 1e6:.2f} MB vs fp32 {mem32 / 1e6:.2f} MB "
          f"({mem32 / mem8:.2f}x less)", flush=True)
    return {"num_params": num_params, "torch_step_ms": tm, "bit_step_ms": bm,
            "speedup": tm / bm, "state_mem_bytes": mem8,
            "fp32_state_mem_bytes": mem32, "memory_ratio": mem32 / mem8}


def parity_case(num_params=262144, steps=8, device="cuda"):
    torch.manual_seed(0)
    p0 = torch.randn(num_params, device=device) * 0.1
    grads = [torch.randn(num_params, device=device) * 0.01 for _ in range(steps)]
    p_t = torch.nn.Parameter(p0.clone())
    p_b = torch.nn.Parameter(p0.clone())
    opt_t = torch.optim.AdamW([p_t], lr=1e-3, weight_decay=0.01)
    opt_b = AdamW8bit([p_b], lr=1e-3, weight_decay=0.01)
    for g in grads:
        p_t.grad = g.clone()
        opt_t.step()
        opt_t.zero_grad()
        p_b.grad = g.clone()
        opt_b.step()
    diff = (p_t.detach() - p_b.detach()).abs()
    rel = (diff.norm() / p_t.detach().norm()).item()
    print(f"parity after {steps} steps: max abs {diff.max().item():.3e}, "
          f"rel {rel:.2e}", flush=True)
    return {"steps": steps, "max_abs_diff": diff.max().item(), "rel_diff": rel}


def train_case(steps=20, device="cuda"):
    torch.manual_seed(0)
    x = torch.randn(256, 512, device=device)
    y = torch.randn(256, 128, device=device)

    def make():
        model = torch.nn.Sequential(
            torch.nn.Linear(512, 512), torch.nn.SiLU(),
            torch.nn.Linear(512, 128),
        ).to(device)
        return model

    def train(use_8bit):
        model = make()
        opt = (AdamW8bit(model.parameters(), lr=1e-3) if use_8bit
               else torch.optim.AdamW(model.parameters(), lr=1e-3))
        losses = []
        t0 = time.perf_counter()
        for _ in range(steps):
            opt.zero_grad()
            out = model(x)
            loss = torch.nn.functional.mse_loss(out, y)
            loss.backward()
            opt.step()
            losses.append(float(loss))
        torch.cuda.synchronize()
        return losses, (time.perf_counter() - t0) / steps * 1000

    l32, ms32 = train(False)
    l8, ms8 = train(True)
    print(f"training {steps} steps: fp32 AdamW {ms32:.2f} ms/step, "
          f"8-bit {ms8:.2f} ms/step ({ms32 / ms8:.2f}x) | final loss "
          f"{l32[-1]:.4f} vs {l8[-1]:.4f}", flush=True)
    return {"steps": steps, "fp32_ms_per_step": ms32, "bit_ms_per_step": ms8,
            "speedup": ms32 / ms8, "final_loss_fp32": l32[-1],
            "final_loss_8bit": l8[-1], "loss_curve_fp32": l32, "loss_curve_8bit": l8}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--vram-fraction", type=float, default=0.85)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        print("CUDA/ZLUDA device is not available")
        return 1
    torch.cuda.set_per_process_memory_fraction(args.vram_fraction)
    print(f"VRAM fraction {args.vram_fraction}", flush=True)
    num = 1024 * 1024 if args.quick else 4 * 1024 * 1024
    report = {
        "step": step_case(num),
        "parity": parity_case(),
        "train": train_case(steps=8 if args.quick else 20),
    }
    save_report("optim8bit", report)
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
