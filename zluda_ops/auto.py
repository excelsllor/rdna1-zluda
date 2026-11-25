import os

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from . import bf16 as _bf16
from . import conv as _conv
from . import flash_attention as _fa
from . import flash_decode as _flash_decode
from . import gemv as _gemv
from . import mha as _mha

_orig_linear = None

_DISPATCH_STATS = {"fa": 0, "decode": 0, "math": 0}
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
        max_m = 16 if input.dtype == torch.float32 else 32
        if (not _needs_grad(input, weight, bias) and input.dim() == 2 and input.shape[0] <= max_m
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
    if (not _compiling() and attn_mask is None and dropout_p == 0.0
            and query.is_cuda and query.dtype == torch.float16
            and query.shape[-1] == 64 and query.shape[2] % 128 == 0):
        try:
            _DISPATCH_STATS["fa"] += 1
            return _fa.scaled_dot_product_attention(
                query, key, value, attn_mask=None, dropout_p=0.0,
                is_causal=is_causal, scale=scale, enable_gqa=enable_gqa,
            )
        except Exception:
            pass
    # Decode: a few query tokens against a long KV cache -> flash-decoding
    # (split-KV), which needs no backward and no mask/scale override.
    if (not _compiling() and attn_mask is None and dropout_p == 0.0
            and scale is None and not enable_gqa
            and query.is_cuda and query.dtype == torch.float16
            and key.dtype == torch.float16 and value.dtype == torch.float16
            and query.dim() == 4 and query.shape[-1] == 64
            and 1 <= query.shape[2] <= 16
            and query.shape[0] == key.shape[0] == value.shape[0]
            and query.shape[1] == key.shape[1] == value.shape[1]
            and key.shape[2] == value.shape[2] and key.shape[2] >= 512
            and key.shape[2] % 128 == 0
            and not _needs_grad(query, key, value)):
        try:
            _DISPATCH_STATS["decode"] += 1
            return _flash_decode.flash_decode_attn(
                query, key, value, causal=bool(is_causal)
            )
        except Exception:
            pass
    _DISPATCH_STATS["math"] += 1
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
    if os.environ.get("ZLUDA_CONV_FP32") == "1":
        _conv.install()
    return True


def uninstall():
    global _orig_linear, _orig_sdpa
    if not installed():
        return False
    F.linear = _orig_linear
    F.scaled_dot_product_attention = _orig_sdpa
    _orig_linear = None
    _orig_sdpa = None
    _conv.uninstall()
    _mha.uninstall()
    return True


if os.environ.get("ZLUDA_MHA_SLOW") == "1":
    _mha.install()

if os.environ.get("ZLUDA_OPS_DISABLE") != "1":
    _installed_now = install()
    if _installed_now and os.environ.get("ZLUDA_AUTO_VERBOSE") == "1":
        print("zluda_ops.auto: patched F.linear and F.scaled_dot_product_attention", flush=True)

__all__ = ["install", "uninstall", "installed"]
