import os

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from . import bf16 as _bf16
from . import gemv as _gemv
from . import flash_attention as _fa

_orig_linear = None
_orig_sdpa = None

_BF16_LIMIT = int(os.environ.get("ZLUDA_AUTO_BF16_MAX_MB", "64")) * 1024 * 1024


def _bf16_allowed(weight):
    return _BF16_LIMIT > 0 and weight.numel() * weight.element_size() <= _BF16_LIMIT


def _compiling():
    try:
        return torch._dynamo.is_compiling()
    except Exception:
        return False


def _needs_grad(*tensors):
    return any(t is not None and isinstance(t, torch.Tensor) and t.requires_grad for t in tensors)


def _linear(input, weight, bias=None):
    if not _compiling() and input.is_cuda:
        if (input.dtype == torch.bfloat16 and weight.dtype == torch.bfloat16
                and input.dim() == 2 and _bf16_allowed(weight) and input.is_contiguous() and weight.is_contiguous()
                and (bias is None or bias.dtype == torch.bfloat16)):
            try:
                return _bf16.linear(input, weight, bias)
            except Exception:
                pass
        if (not _needs_grad(input, weight, bias) and input.dim() == 2 and input.shape[0] <= 8
                and input.dtype in (torch.float16, torch.float32)
                and weight.dtype == input.dtype and _gemv._supported(input, weight, bias)):
            try:
                out = _gemv.linear(input, weight, bias, accurate=True)
                if out is not None:
                    return out
            except Exception:
                pass
    return _orig_linear(input, weight, bias)


def _sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False,
          scale=None, enable_gqa=False):
    if (not _compiling() and is_causal and attn_mask is None and dropout_p == 0.0
            and query.is_cuda and query.dtype == torch.float16
            and query.shape[-1] == 64 and query.shape[2] % 128 == 0):
        try:
            return _fa.scaled_dot_product_attention(
                query, key, value, attn_mask=None, dropout_p=0.0,
                is_causal=True, scale=scale, enable_gqa=enable_gqa,
            )
        except Exception:
            pass
    with sdpa_kernel(SDPBackend.MATH):
        return _orig_sdpa(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p,
                          is_causal=is_causal, scale=scale, enable_gqa=enable_gqa)


def installed():
    return _orig_linear is not None


def install():
    global _orig_linear, _orig_sdpa
    if installed() or os.environ.get("ZLUDA_OPS_DISABLE") == "1":
        return False
    _orig_linear = F.linear
    _orig_sdpa = F.scaled_dot_product_attention
    F.linear = _linear
    F.scaled_dot_product_attention = _sdpa
    return True


def uninstall():
    global _orig_linear, _orig_sdpa
    if not installed():
        return False
    F.linear = _orig_linear
    F.scaled_dot_product_attention = _orig_sdpa
    _orig_linear = None
    _orig_sdpa = None
    return True


if os.environ.get("ZLUDA_OPS_DISABLE") != "1":
    _installed_now = install()
    if _installed_now and os.environ.get("ZLUDA_AUTO_VERBOSE") == "1":
        print("zluda_ops.auto: patched F.linear and F.scaled_dot_product_attention", flush=True)

__all__ = ["install", "uninstall", "installed"]
