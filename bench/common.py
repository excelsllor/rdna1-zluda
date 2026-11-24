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


def gpu_info():
    props = torch.cuda.get_device_properties(0)
    info = {
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "hip": getattr(torch.version, "hip", None),
        "multi_processor_count": props.multi_processor_count,
        "total_memory_gb": round(props.total_memory / 1024**3, 2),
        "l2_cache_mb": round(getattr(props, "L2_cache_size", 0) / 1024**2, 2),
        "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "hip_path": os.environ.get("HIP_PATH"),
        "zluda_cc": os.environ.get("ZLUDA_CC"),
    }
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
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"report: {path}")
    return path
