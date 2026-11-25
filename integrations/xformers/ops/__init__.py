"""xformers.ops shim: memory-efficient attention via ZLUDA Triton kernels."""

import torch
import torch.nn.functional as F


class AttentionBias:
    """Base class for xformers attention biases."""


class LowerTriangularMask(AttentionBias):
    """Causal mask marker (maps to is_causal=True)."""


class TensorBias(AttentionBias):
    """Additive bias; `bias` may be broadcastable to (B, H, S, S) or (S, S)."""

    def __init__(self, bias):
        self.bias = bias


class BlockDiagonalMask(AttentionBias):
    """Block-diagonal (varlen) bias; handled by the math fallback if needed."""

    def __init__(self, q_seqlen, kv_seqlen=None):
        self.q_seqlen = q_seqlen
        self.kv_seqlen = kv_seqlen if kv_seqlen is not None else q_seqlen


class BlockDiagonalCausalMask(BlockDiagonalMask):
    pass


def _to_bhsd(t):
    return t.transpose(1, 2).contiguous()


def _bias_to_mask(attn_bias, q, k):
    if attn_bias is None:
        return None, False
    if isinstance(attn_bias, LowerTriangularMask):
        return None, True
    if isinstance(attn_bias, TensorBias):
        bias = attn_bias.bias
        if bias.dtype != q.dtype:
            bias = bias.to(q.dtype)
        return bias, False
    return None, False


def memory_efficient_attention_forward(query, key, value, attn_bias=None,
                                       p=0.0, scale=None, output_dtype=None):
    """xformers API: (B, S, H, D) tensors -> (B, S, H, D) tensor."""
    if output_dtype is None:
        output_dtype = query.dtype
    q_bhsd, k_bhsd, v_bhsd = _to_bhsd(query), _to_bhsd(key), _to_bhsd(value)
    mask, causal = _bias_to_mask(attn_bias, query, key)
    try:
        from zluda_ops.flash_attention import scaled_dot_product_attention as _fa

        out = _fa(q_bhsd, k_bhsd, v_bhsd, attn_mask=mask, dropout_p=p,
                  is_causal=causal, scale=scale)
    except Exception:  # noqa: BLE001 - fall back to math SDPA
        from torch.nn.attention import SDPBackend, sdpa_kernel

        with sdpa_kernel(SDPBackend.MATH):
            out = F.scaled_dot_product_attention(
                q_bhsd, k_bhsd, v_bhsd, attn_mask=mask, dropout_p=p,
                is_causal=causal, scale=scale)
    return out.transpose(1, 2).contiguous().to(output_dtype)


def memory_efficient_attention(query, key, value, attn_bias=None, p=0.0,
                               scale=None, output_dtype=None):
    return memory_efficient_attention_forward(query, key, value, attn_bias, p,
                                              scale, output_dtype)


def unbind(x, dim=0):
    return x.unbind(dim)


class fmha:
    """Minimal `xformers.ops.fmha` namespace used by some integrations."""

    memory_efficient_attention = staticmethod(memory_efficient_attention)
    memory_efficient_attention_forward = staticmethod(
        memory_efficient_attention_forward)
    AttentionBias = AttentionBias
    LowerTriangularMask = LowerTriangularMask
    TensorBias = TensorBias
    BlockDiagonalMask = BlockDiagonalMask
    BlockDiagonalCausalMask = BlockDiagonalCausalMask


__all__ = [
    "memory_efficient_attention", "memory_efficient_attention_forward",
    "AttentionBias", "LowerTriangularMask", "TensorBias", "BlockDiagonalMask",
    "BlockDiagonalCausalMask", "fmha", "unbind",
]
