"""Per-shape solver selection for conv2d on gfx1010 (MIOpen).

MIOpen uses different solvers depending on the data type and shape:

* fp32 3x3 convolutions -> `ConvBinWinogradRxSf2x3g1` (fast Winograd);
* fp16 3x3 -> `GemmFwdRest` (rocBLAS GEMM), which is 1.4-2.7x slower forward
  and up to 1.9x slower forward+backward on SD/ResNet shapes;
* for some small shapes the fp16 path is nevertheless faster than the
  fp32-cast workaround because the casts dominate.

This module measures both candidates once per shape and caches the winner
(`_SOLVER_CACHE`), and caches the fp32 copies of the weights (`_W_CACHE`) so the
fp32 path only pays for the input cast on every call. Everything is
transparent: the fp32 path casts the inputs, runs the fp32 Winograd
convolution and casts the result back (fp16 inputs are exactly representable in
fp32, so the result equals an fp32 convolution rounded to fp16).

Enable either with:

  ZLUDA_CONV_AUTO=1   per-shape auto-selection (recommended)
  ZLUDA_CONV_FP32=1   force the fp32 Winograd path for eligible shapes

or call `conv2d(..., fp32_winograd=False/True)` / `install()` directly.
"""

import os
from collections import OrderedDict

import torch
import torch.nn.functional as F

_orig_conv2d = None
# key: (weight data_ptr, weight._version, shape, dtype) -> (w32, b32, weight_ref)
_W_CACHE = OrderedDict()
_W_CACHE_MAX = 256
# key: (x.shape, weight.shape, stride, padding, dilation, groups) -> "fp16"|"fp32"
_SOLVER_CACHE = {}
_SOLVER_BENCH_ITERS = 3


def _normalize_pair(value):
    if isinstance(value, (tuple, list)):
        return tuple(value)
    return (value, value)


def is_eligible(x, weight, stride, padding, dilation, groups):
    if not (x.is_cuda and x.dtype == torch.float16 and weight.dtype == torch.float16):
        return False
    if not (x.is_contiguous() and weight.is_contiguous()):
        return False
    if weight.dim() != 4 or weight.shape[-2:] != (3, 3):
        return False
    if _normalize_pair(stride) != (1, 1) or _normalize_pair(dilation) != (1, 1):
        return False
    if groups != 1:
        return False
    return True


def _shape_key(x, weight, bias, stride, padding, dilation, groups):
    return (tuple(x.shape), tuple(weight.shape),
            None if bias is None else tuple(bias.shape),
            _normalize_pair(stride), _normalize_pair(padding),
            _normalize_pair(dilation), groups)


def _fp32_params(weight, bias):
    key = (weight.data_ptr(), weight._version, tuple(weight.shape), weight.dtype)
    cached = _W_CACHE.get(key)
    if cached is not None:
        _W_CACHE.move_to_end(key)
        return cached[0], cached[1]
    w32 = weight.detach().float()
    b32 = None if bias is None else bias.detach().float()
    _W_CACHE[key] = (w32, b32, weight)  # keep a reference so data_ptr stays valid
    while len(_W_CACHE) > _W_CACHE_MAX:
        _W_CACHE.popitem(last=False)
    return w32, b32


def _conv_fp32_cached(x, weight, bias, stride, padding):
    w32, b32 = _fp32_params(weight, bias)
    y = F.conv2d(x.float(), w32, b32, stride=stride, padding=padding)
    return y.to(x.dtype)


def _conv_fp16(x, weight, bias, stride, padding, dilation, groups):
    original = _orig_conv2d if _orig_conv2d is not None else F.conv2d
    return original(x, weight, bias, stride=stride, padding=padding,
                    dilation=dilation, groups=groups)


def _time(fn, iters=_SOLVER_BENCH_ITERS):
    torch.cuda.synchronize()
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def _choose_solver(x, weight, bias, stride, padding, dilation, groups):
    key = _shape_key(x, weight, bias, stride, padding, dilation, groups)
    chosen = _SOLVER_CACHE.get(key)
    if chosen is not None:
        return chosen
    try:
        t16 = _time(lambda: _conv_fp16(x, weight, bias, stride, padding,
                                       dilation, groups))
        t32 = _time(lambda: _conv_fp32_cached(x, weight, bias, stride, padding))
        chosen = "fp16" if t16 <= t32 else "fp32"
        _SOLVER_CACHE[key] = chosen
    except Exception:
        chosen = "fp32"
        _SOLVER_CACHE[key] = chosen
    return chosen


def conv2d(x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1,
           fp32_winograd=None, auto=None):
    """Drop-in conv2d with per-shape solver selection for eligible fp16 3x3."""
    if auto is None:
        auto = os.environ.get("ZLUDA_CONV_AUTO") == "1"
    if fp32_winograd is None:
        fp32_winograd = os.environ.get("ZLUDA_CONV_FP32") == "1"
    eligible = is_eligible(x, weight, stride, padding, dilation, groups)
    if not eligible:
        return _conv_fp16(x, weight, bias, stride, padding, dilation, groups)
    if fp32_winograd:
        return _conv_fp32_cached(x, weight, bias, stride, padding)
    if auto:
        chosen = _choose_solver(x, weight, bias, stride, padding, dilation, groups)
        if chosen == "fp32":
            return _conv_fp32_cached(x, weight, bias, stride, padding)
    return _conv_fp16(x, weight, bias, stride, padding, dilation, groups)


def install():
    """Patch F.conv2d (auto or forced-fp32 mode, controlled by env vars)."""
    global _orig_conv2d
    if _orig_conv2d is not None:
        return
    _orig_conv2d = F.conv2d

    def _conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1,
                groups=1):
        return conv2d(input, weight, bias, stride=stride, padding=padding,
                      dilation=dilation, groups=groups)

    F.conv2d = _conv2d


def uninstall():
    global _orig_conv2d
    if _orig_conv2d is None:
        return
    F.conv2d = _orig_conv2d
    _orig_conv2d = None


def installed():
    return _orig_conv2d is not None


def stats():
    return {"weight_cache": len(_W_CACHE), "solver_cache": len(_SOLVER_CACHE),
            "choices": dict(_SOLVER_CACHE)}


def clear_caches():
    _W_CACHE.clear()
    _SOLVER_CACHE.clear()


__all__ = ["conv2d", "install", "uninstall", "installed", "is_eligible",
           "stats", "clear_caches"]
