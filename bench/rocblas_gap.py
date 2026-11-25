"""Check whether the fp32 GEMM gap between PyTorch and direct rocBLAS is real.

Older measurements disagreed: `bench/stage1.py` reports 8.16 TF for a 4096^2
fp32 `torch.matmul`, while isolated ctypes scripts reported 9.2-9.6 TF for
direct `rocblas_sgemm`/`rocblas_gemm_ex` calls. This script runs every variant
in one process with a warmed GPU, shuffled rounds and median timing, and adds a
zero-data control:

  * all call paths (torch mm, torch mm with out=, direct rocBLAS on torch
    buffers, direct rocBLAS on hipMalloc buffers) agree on random data;
  * the same direct call on zeroed buffers is ~12% faster, which explains the
    old 9.2-9.6 TF numbers: those scripts used uninitialised `hipMalloc`
    buffers. Zero data draws less power, so the GPU holds higher clocks; it is
    not a software gap and nothing in ZLUDA/torch can recover it.

Conclusion: use random data and median timing everywhere; 8.16-8.20 TF is the
representative fp32 4096^2 result on this card.

Usage:
  run-zluda.cmd venv\\Scripts\\python.exe bench\\rocblas_gap.py --n 2048
  run-zluda.cmd venv\\Scripts\\python.exe bench\\rocblas_gap.py --n 4096 --rounds 5
  DISABLE_ADDMM_CUDA_LT=1 run-zluda.cmd venv\\Scripts\\python.exe bench\\rocblas_gap.py --n 4096

The CUDA-Lt vs cuBLAS choice is a process-wide environment variable, so run
the script twice with and without DISABLE_ADDMM_CUDA_LT=1.
"""

import argparse
import ctypes
import os
import random
import statistics
import sys
import time
from pathlib import Path

import torch

from common import save_report

REPO = Path(__file__).resolve().parents[1]

ROCBLAS_OP_N = 111
HIP_MEMCPY_D2D = 3


def load_rocblas():
    hip_path = os.environ.get("HIP_PATH")
    if not hip_path:
        hip_path = str(REPO / "hip")
    hip_bin = Path(hip_path) / "bin"
    if hasattr(os, "add_dll_directory"):
        try:
            os.add_dll_directory(str(hip_bin))
        except OSError:
            pass
    hip = ctypes.CDLL(str(hip_bin / "amdhip64_7.dll"))
    rb = ctypes.CDLL(str(hip_bin / "rocblas.dll"))

    hip.hipInit.argtypes = [ctypes.c_uint]
    hip.hipInit.restype = ctypes.c_int
    hip.hipMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
    hip.hipMalloc.restype = ctypes.c_int
    hip.hipFree.argtypes = [ctypes.c_void_p]
    hip.hipFree.restype = ctypes.c_int
    hip.hipMemcpy.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int
    ]
    hip.hipMemcpy.restype = ctypes.c_int
    hip.hipMemset.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
    hip.hipMemset.restype = ctypes.c_int
    hip.hipDeviceSynchronize.argtypes = []
    hip.hipDeviceSynchronize.restype = ctypes.c_int

    rb.rocblas_create_handle.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    rb.rocblas_create_handle.restype = ctypes.c_int
    rb.rocblas_sgemm.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
        ctypes.c_void_p, ctypes.c_int,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
    ]
    rb.rocblas_sgemm.restype = ctypes.c_int
    return hip, rb


def make_sgemm(rb, handle, alpha, beta, a_ptr, b_ptr, c_ptr, n):
    # Row-major C = A@B is computed by a column-major call with B and A
    # swapped (square buffers).
    def call():
        return rb.rocblas_sgemm(
            handle, ROCBLAS_OP_N, ROCBLAS_OP_N, n, n, n,
            ctypes.byref(alpha), ctypes.c_void_p(b_ptr), n,
            ctypes.c_void_p(a_ptr), n,
            ctypes.byref(beta), ctypes.c_void_p(c_ptr), n,
        )
    return call


def time_call(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters * 1000.0


def tflops(n, ms):
    return 2.0 * n ** 3 / (ms / 1000.0) / 1e12


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=2048)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--vram-fraction", type=float, default=None,
                        help="cap the PyTorch allocator (e.g. 0.8) for risky runs")
    args = parser.parse_args()

    if args.vram_fraction:
        torch.cuda.set_per_process_memory_fraction(args.vram_fraction)
        print(f"VRAM fraction {args.vram_fraction}", flush=True)

    n = args.n
    torch.manual_seed(0)
    a = torch.randn(n, n, device="cuda", dtype=torch.float32)
    b = torch.randn(n, n, device="cuda", dtype=torch.float32)
    c = torch.empty(n, n, device="cuda", dtype=torch.float32)
    torch.cuda.synchronize()

    hip, rb = load_rocblas()
    assert hip.hipInit(0) == 0
    handle = ctypes.c_void_p()
    assert rb.rocblas_create_handle(ctypes.byref(handle)) == 0
    alpha = ctypes.c_float(1.0)
    beta = ctypes.c_float(0.0)

    # Hip-allocated buffers, initialised from the torch tensors (same content
    # so compression does not skew the comparison).
    hip_ptrs, hip_bufs = [], []
    nbytes = n * n * 4
    for src in (a, b, c):
        ptr = ctypes.c_void_p()
        assert hip.hipMalloc(ctypes.byref(ptr), nbytes) == 0
        assert hip.hipMemcpy(ptr, ctypes.c_void_p(src.data_ptr()), nbytes,
                             HIP_MEMCPY_D2D) == 0
        hip_ptrs.append(ptr)
        hip_bufs.append(ptr)
    hip.hipDeviceSynchronize()

    sgemm_torch = make_sgemm(rb, handle, alpha, beta,
                             a.data_ptr(), b.data_ptr(), c.data_ptr(), n)
    sgemm_hip = make_sgemm(rb, handle, alpha, beta,
                           hip_ptrs[0].value, hip_ptrs[1].value,
                           hip_ptrs[2].value, n)
    assert sgemm_torch() == 0 and sgemm_hip() == 0
    torch.cuda.synchronize()

    # Zero-data control: same direct call, same allocations, zeroed content.
    # This is what the old isolated scripts accidentally measured.
    zero_ptrs = []
    for _ in range(3):
        ptr = ctypes.c_void_p()
        assert hip.hipMalloc(ctypes.byref(ptr), nbytes) == 0
        assert hip.hipMemset(ptr, 0, nbytes) == 0
        zero_ptrs.append(ptr)
    sgemm_hip_zeros = make_sgemm(rb, handle, alpha, beta,
                                 zero_ptrs[0].value, zero_ptrs[1].value,
                                 zero_ptrs[2].value, n)
    assert sgemm_hip_zeros() == 0
    torch.cuda.synchronize()

    # Sanity: direct call on torch buffers must match torch.mm.
    ref = a @ b
    err = (c - ref).abs().max().item()
    del ref
    torch.cuda.synchronize()

    variants = {
        "torch_mm": lambda: torch.mm(a, b),
        "torch_mm_out": lambda: torch.mm(a, b, out=c),
        "sgemm_torch_buf": sgemm_torch,
        "sgemm_hip_buf": sgemm_hip,
        "sgemm_hip_zeros": sgemm_hip_zeros,
    }

    samples = {name: [] for name in variants}
    order = list(variants)
    rng = random.Random(1234)
    for _ in range(args.rounds):
        rng.shuffle(order)
        for name in order:
            samples[name].append(time_call(variants[name], args.warmup, args.iters))

    rows = []
    print(f"n={n} rounds={args.rounds} iters={args.iters} "
          f"tf32={torch.backends.cuda.matmul.allow_tf32} "
          f"precision={torch.get_float32_matmul_precision()} "
          f"addmm_lt_disabled={os.environ.get('DISABLE_ADDMM_CUDA_LT')}", flush=True)
    print(f"direct sgemm max abs error vs torch.mm: {err:.3e}", flush=True)
    for name, values in samples.items():
        med = statistics.median(values)
        best = min(values)
        rows.append({
            "variant": name,
            "median_ms": med,
            "best_ms": best,
            "median_tflops": tflops(n, med),
            "best_tflops": tflops(n, best),
            "samples_ms": values,
        })
        print(f"{name:16s} median {med:8.3f} ms {tflops(n, med):6.2f} TF | "
              f"best {best:8.3f} ms {tflops(n, best):6.2f} TF", flush=True)

    for ptr in hip_bufs + zero_ptrs:
        hip.hipFree(ptr)
    save_report("rocblas_gap", {
        "n": n,
        "rounds": args.rounds,
        "iters": args.iters,
        "direct_error": err,
        "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "disable_addmm_cuda_lt": os.environ.get("DISABLE_ADDMM_CUDA_LT"),
        "vram_fraction": args.vram_fraction,
        "rows": rows,
    })


if __name__ == "__main__":
    sys.exit(main())
