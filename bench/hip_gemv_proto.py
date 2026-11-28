"""HIP GEMV prototype harness (audit correction #9: the docs referenced a
harness that did not exist).

The prototype DLL is built from tools/hip_proto/gemv_fp16.cpp with
tools/hip_proto/build_gemv.cmd and exports
    int hip_gemv_bench(void* x, void* w, void* y, int N, int K, int iters)
It is called through ctypes and compared against the Triton GEMV
(zluda_ops.gemv.linear) on the same buffers (under ZLUDA cudaMalloc is hipMalloc,
so a torch tensor pointer is a valid HIP device pointer for the prototype).

Run under run-zluda.cmd:
  venv\\Scripts\\python.exe bench\\hip_gemv_proto.py [--dll PATH] [--n 151936] [--k 896]
"""

import argparse
import ctypes
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench.common import save_report  # noqa: E402

CANDIDATES = [
    Path(__import__("os").environ.get("TEMP", ".")) / "hip_gemv4.dll",
    Path(__import__("os").environ.get("TEMP", ".")) / "hip_gemv.dll",
    Path(__file__).resolve().parents[1] / "tools" / "hip_proto" / "build" / "hip_gemv.dll",
]


def find_dll(explicit):
    if explicit:
        p = Path(explicit)
        return p if p.exists() else None
    for p in CANDIDATES:
        if p.exists():
            return p
    return None


def bench(fn, n=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1e6


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dll", default=None)
    parser.add_argument("--n", type=int, default=151936)
    parser.add_argument("--k", type=int, default=896)
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        print("CUDA/ZLUDA device is not available")
        return 1
    torch.cuda.set_per_process_memory_fraction(0.7)
    from zluda_ops import gemv as G

    dll = find_dll(args.dll)
    result = {"n": args.n, "k": args.k, "dll": str(dll) if dll else None}
    if dll is None:
        result["status"] = "skipped"
        result["hint"] = ("build it with tools\\hip_proto\\build_gemv.cmd "
                          "(needs ROCm clang++) and pass --dll")
        save_report("hip_gemv_proto", result)
        print(f"SKIP: prototype DLL not found; {result['hint']}")
        return 0
    fn = ctypes.CDLL(str(dll)).hip_gemv_bench
    fn.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int, ctypes.c_int, ctypes.c_int]
    fn.restype = ctypes.c_int
    x = torch.randn(args.k, device="cuda", dtype=torch.float16)
    w = torch.randn(args.n, args.k, device="cuda", dtype=torch.float16)
    y = torch.empty(args.n, device="cuda", dtype=torch.float16)

    def hip_run():
        rc = fn(x.data_ptr(), w.data_ptr(), y.data_ptr(), args.n, args.k,
                args.iters)
        if rc:
            raise RuntimeError(f"hip_gemv_bench rc={rc}")
        torch.cuda.synchronize()

    def triton_run():
        G.linear(x.unsqueeze(0), w, None, accurate=True)

    hip_us = None
    try:
        # the prototype loops --iters internally per call
        t0 = time.perf_counter()
        hip_run()
        hip_us = (time.perf_counter() - t0) * 1e6 / args.iters
        result["hip_us"] = round(hip_us, 1)
        result["hip_status"] = "OK"
    except Exception as exc:  # noqa: BLE001
        result["hip_status"] = f"FAIL {type(exc).__name__}: {exc}"
    triton_us = bench(triton_run, n=max(20, args.iters)) if torch.cuda.is_available() else None
    result["triton_us"] = round(triton_us, 1) if triton_us else None
    if hip_us and triton_us:
        result["speedup_vs_triton"] = round(triton_us / hip_us, 2)
    bytes_read = args.n * args.k * 2
    result["hip_GBs"] = round(bytes_read / hip_us / 1e3, 1) if hip_us else None
    result["triton_GBs"] = round(bytes_read / triton_us / 1e3, 1) if triton_us else None
    save_report("hip_gemv_proto", result)
    print(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
