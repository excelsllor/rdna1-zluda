import argparse
import json
import subprocess
import time

import torch

from common import nvml_sample, save_report

GPU_COUNTER = (
    "(Get-Counter '\\GPU Engine(*)\\Utilization Percentage' -ErrorAction SilentlyContinue)"
    ".CounterSamples | Measure-Object CookedValue -Sum | Select-Object -ExpandProperty Sum"
)


def utilization():
    try:
        proc = subprocess.run(
            ["powershell", "-NoProfile", "-Command", GPU_COUNTER],
            capture_output=True, text=True, timeout=5,
        )
        value = proc.stdout.strip().replace(",", ".")
        if value:
            return max(0.0, min(100.0, float(value)))
    except Exception:
        pass
    try:
        return float(torch.cuda.utilization(0))
    except Exception:
        return None


def sample():
    free, total = torch.cuda.mem_get_info()
    return {
        "time": time.time(),
        "gpu_util_pct": utilization(),
        "vram_free_mb": free / 1024**2,
        "vram_total_mb": total / 1024**2,
        "vram_used_mb": (total - free) / 1024**2,
        "torch_allocated_mb": torch.cuda.memory_allocated() / 1024**2,
        "torch_reserved_mb": torch.cuda.memory_reserved() / 1024**2,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--interval", type=float, default=1.0)
    args = parser.parse_args()
    if torch.cuda.is_available():
        torch.cuda.init()
    rows = []
    end = time.time() + args.seconds
    while time.time() < end:
        rows.append(sample())
        time.sleep(args.interval)
    usable = [r["gpu_util_pct"] for r in rows if r["gpu_util_pct"] is not None]
    report = {
        "samples": rows,
        "gpu_util_avg_pct": sum(usable) / len(usable) if usable else None,
        "gpu_util_max_pct": max(usable) if usable else None,
        "clock_power": nvml_sample(),
        "limitations": [
            "gfx1010 Windows has no rocm-smi/amd-smi; ZLUDA NVML reports NotSupported",
            "GPU utilization comes from Windows performance counters and includes other processes",
            "sclk/mclk/temperature/power are unavailable",
        ],
    }
    print(json.dumps(report, indent=2))
    save_report("monitor", report)


if __name__ == "__main__":
    main()
