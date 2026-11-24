import warnings
from contextlib import contextmanager

import torch

_ENABLED = False


def available():
    return hasattr(torch._C, "_get_cublas_allow_fp16_accumulation")


def enabled():
    if not available():
        return False
    return bool(torch._C._get_cublas_allow_fp16_accumulation())


def describe():
    return (
        "fp16 accumulation makes cuBLAS/rocBLAS accumulate GEMM products in "
        "fp16 instead of fp32. It enables packed FP16 kernels on gfx1010 "
        "(roughly 2x inference GEMM throughput) but reduces numerical "
        "accuracy. Use only for inference and validate your model."
    )


@contextmanager
def fp16_accumulation(enable=True, warn=True):
    if not available():
        raise RuntimeError("torch._C fp16 accumulation control is unavailable")
    previous = enabled()
    if enable and warn:
        warnings.warn(describe(), RuntimeWarning, stacklevel=2)
    torch._C._set_cublas_allow_fp16_accumulation(bool(enable))
    try:
        yield
    finally:
        torch._C._set_cublas_allow_fp16_accumulation(previous)


def set_fp16_accumulation(enable, allow_unsafe=False):
    if enable and not allow_unsafe:
        raise ValueError(
            "fp16 accumulation is numerically unsafe; pass allow_unsafe=True "
            "or use the fp16_accumulation() context manager"
        )
    if not available():
        raise RuntimeError("torch._C fp16 accumulation control is unavailable")
    torch._C._set_cublas_allow_fp16_accumulation(bool(enable))


__all__ = [
    "available", "enabled", "describe", "fp16_accumulation",
    "set_fp16_accumulation",
]
