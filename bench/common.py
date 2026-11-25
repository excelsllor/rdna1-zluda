import json
import os
import time
from datetime import datetime
from pathlib import Path

import torch


def bench(fn, warmup=3, iters=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters * 1000.0


def bench_median(fn, rounds=5, iters=5, warmup=2):
    """Median of per-round mean times; use for the reported RDNA1 numbers.

    Rounds are separated so a single clock/power excursion does not leak into
    the result. Benchmarks that compare variants should also shuffle variant
    order between rounds (see bench/rocblas_gap.py).
    """
    samples = []
    for _ in range(rounds):
        samples.append(bench(fn, warmup=warmup, iters=iters))
    samples.sort()
    mid = len(samples) // 2
    if len(samples) % 2:
        return samples[mid]
    return (samples[mid - 1] + samples[mid]) / 2.0


def gpu_info():
    info = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "hip": getattr(torch.version, "hip", None),
        "hip_path": os.environ.get("HIP_PATH"),
        "zluda_cc": os.environ.get("ZLUDA_CC"),
    }
    try:
        props = torch.cuda.get_device_properties(0)
        info.update({
            "device": torch.cuda.get_device_name(0),
            "multi_processor_count": props.multi_processor_count,
            "total_memory_gb": round(props.total_memory / 1024**3, 2),
            "l2_cache_mb": round(getattr(props, "L2_cache_size", 0) / 1024**2, 2),
        })
    except Exception as exc:  # plain python without ZLUDA injection
        info["device"] = None
        info["device_error"] = f"{type(exc).__name__}: {str(exc)[:160]}"
    info["allow_tf32"] = torch.backends.cuda.matmul.allow_tf32
    info["float32_matmul_precision"] = torch.get_float32_matmul_precision()
    try:
        import triton

        info["triton"] = triton.__version__
    except Exception as exc:
        info["triton"] = f"unavailable: {type(exc).__name__}"
    return info


def nvml_sample():
    result = {}
    for name in ("utilization", "temperature", "clock_rate", "power_draw", "memory_usage"):
        fn = getattr(torch.cuda, name, None)
        if fn is None:
            result[name] = None
            continue
        try:
            value = fn(0)
            if isinstance(value, tuple):
                value = list(value)
            result[name] = value
        except Exception as exc:
            result[name] = None
            result.setdefault("errors", {})[name] = f"{type(exc).__name__}: {str(exc)[:120]}"
    return result


def save_report(name, payload):
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = out_dir / f"{name}-{stamp}.json"
    payload = dict(payload)
    payload["timestamp"] = datetime.now().isoformat(timespec="seconds")
    payload["gpu"] = gpu_info()
    # Every report records how it was measured: zeros/ones/uninitialised data
    # inflate GEMM by ~12% (bench/rocblas_gap.py), and clocks cannot be fixed
    # because ADL live status is NOT_SUPPORTED on this driver.
    payload.setdefault("protocol", {
        "data": "random data (torch.randn) unless stated otherwise",
        "timing": "mean per round, median across rounds, warmup per round",
        "order": "variant order shuffled between rounds when comparing",
        "clocks": "unfixed; sclk/mclk unavailable (ADL NOT_SUPPORTED, see bench/gpu_clocks.py)",
    })
    try:
        from gpu_clocks import sample as clocks_sample

        payload["clocks"] = clocks_sample()
    except Exception as exc:  # optional telemetry must never fail a report
        payload["clocks"] = {"errors": {"gpu_clocks": f"{type(exc).__name__}: {exc}"}}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"report: {path}")
    return path
