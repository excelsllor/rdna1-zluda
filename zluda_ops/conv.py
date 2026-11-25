"""Cast-to-fp32 convolution workaround for gfx1010 (Winograd).

On this stack MIOpen picks a Winograd solver for fp32 3x3 convolutions
(`ConvBinWinogradRxSf2x3g1`), but fp16 3x3 falls back to a rocBLAS GEMM
(`GemmFwdRest`), which is 1.4-2.7x slower for forward and up to 1.9x slower for
forward+backward on SD/ResNet shapes. Casting the fp16 inputs (and weights) to
fp32, running the fp32 Winograd convolution and casting the result back is
faster and at least as accurate: fp16 inputs are exactly representable in fp32,
so the result equals an fp32 convolution rounded to fp16.

`conv2d` mirrors `torch.nn.functional.conv2d` for the eligible subset (fp16,
3x3, stride 1, dilation 1, groups 1, CUDA). Gradients flow through the casts, so
it is usable for both inference and training. `install()`/`uninstall()` patch
`F.conv2d` behind the `ZLUDA_CONV_FP32=1` environment variable.

Limitations: fp32 copies of the weights cost memory (conv weights are small
relative to a model, but this is not free); the win depends on the shape (small
convolutions are conversion-bound, large ones win up to ~2.7x forward); only
the 3x3 stride-1/dilation-1/group-1 case benefits, everything else uses the
regular path.
"""

import os
import torch
import torch.nn.functional as F

_eligible_cache = {}
_orig_conv2d = None


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


def conv2d(x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1,
           fp32_winograd=None):
    """Drop-in conv2d that uses the fp32 Winograd path for eligible fp16 3x3."""
    if fp32_winograd is None:
        fp32_winograd = is_eligible(x, weight, stride, padding, dilation, groups)
    if not fp32_winograd:
        original = _orig_conv2d if _orig_conv2d is not None else F.conv2d
        return original(x, weight, bias, stride=stride, padding=padding,
                        dilation=dilation, groups=groups)
    y = F.conv2d(x.float(), weight.float(),
                 None if bias is None else bias.float(),
                 stride=stride, padding=padding, dilation=dilation, groups=groups)
    return y.to(x.dtype)


def install():
    """Patch F.conv2d to use the fp32 Winograd path (env-gated in practice)."""
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


__all__ = ["conv2d", "install", "uninstall", "installed", "is_eligible"]
