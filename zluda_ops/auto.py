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



_SDPA_FLAGS = None


def _disable_fast_sdpa():
    """Disable PyTorch SDPA backends that crash under ZLUDA.

    The flash/efficient/cudnn SDPA backends either report "no available kernel"
    or fail with an async launch error that poisons the CUDA context (a crash in
    `fmha` kernels). MATH is the only safe backend; our Triton FA/flash-decoding
    paths are dispatched by this module before falling back to math.
    """
    global _SDPA_FLAGS
    try:
        import torch

        _SDPA_FLAGS = (
            torch.backends.cuda.flash_sdp_enabled(),
            torch.backends.cuda.mem_efficient_sdp_enabled(),
            torch.backends.cuda.cudnn_sdp_enabled(),
        )
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_cudnn_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    except Exception:
        pass


def _restore_sdpa():
    global _SDPA_FLAGS
    if _SDPA_FLAGS is None:
        return
    try:
        import torch

        flash, eff, cudnn = _SDPA_FLAGS
        torch.backends.cuda.enable_flash_sdp(flash)
        torch.backends.cuda.enable_mem_efficient_sdp(eff)
        torch.backends.cuda.enable_cudnn_sdp(cudnn)
    except Exception:
        pass
    _SDPA_FLAGS = None

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
        grad_needed = torch.is_grad_enabled() and _needs_grad(input, weight, bias)
        # Decode calls F.linear with (B, S, D); flatten B=1 small-S inputs so
        # the Triton GEMV path can serve token-by-token inference.
        flat = input
        out_shape = None
        if (input.dim() == 3 and input.shape[0] == 1 and input.shape[1] <= max_m):
            flat = input.reshape(-1, input.shape[-1])
            out_shape = input.shape[:-1] + (weight.shape[0],)
        if (not grad_needed and flat.dim() == 2 and flat.shape[0] <= max_m
                and flat.dtype in (torch.float16, torch.float32)
                and weight.dtype == flat.dtype and _gemv._supported(flat, weight, bias)):
            try:
                out = _gemv.linear(flat, weight, bias, accurate=True)
                if out is not None:
                    return out.reshape(out_shape) if out_shape is not None else out
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
            and scale is None
            and query.is_cuda and query.dtype == torch.float16
            and key.dtype == torch.float16 and value.dtype == torch.float16
            and query.dim() == 4 and query.shape[-1] == 64
            and 1 <= query.shape[2] <= 16
            and query.shape[0] == key.shape[0] == value.shape[0]
            and key.shape[1] == value.shape[1]
            and query.shape[1] % key.shape[1] == 0
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
    _disable_fast_sdpa()
    if (os.environ.get("ZLUDA_CONV_FP32") == "1"
            or os.environ.get("ZLUDA_CONV_AUTO") == "1"):
        _conv.install()
    return True


if os.environ.get("ZLUDA_MHA_SLOW") == "1":
    _mha.install()

if os.environ.get("ZLUDA_OPS_DISABLE") != "1":
    _installed_now = install()
    if _installed_now and os.environ.get("ZLUDA_AUTO_VERBOSE") == "1":
        print("zluda_ops.auto: patched F.linear and F.scaled_dot_product_attention", flush=True)

__all__ = ["install", "uninstall", "installed"]
