"""flash_attn-compatible shim backed by the ZLUDA Triton kernels.

pip install flash-attn cannot work on this stack (no nvcc, no matrix cores).
This module provides the public surface used by transformers/inference code,
implemented with zluda_ops: Triton flash-attention for prefill, split-KV
flash-decoding for decode (including GQA), math SDPA as the fallback.

Layouts follow the real package: (B, S, H, D). Install with
`tools/install_ecosystem.py`.
"""

import torch
import torch.nn.functional as F

__version__ = "2.7.4+zluda-shim"
_SHIM_ORIG_SDPA = F.scaled_dot_product_attention


def _math_sdpa(q, k, v, dropout_p, scale, causal):
    from torch.nn.attention import SDPBackend, sdpa_kernel

    with sdpa_kernel(SDPBackend.MATH):
        return _SHIM_ORIG_SDPA(q, k, v, attn_mask=None, dropout_p=dropout_p,
                               is_causal=causal, scale=scale)


def flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False,
                    window_size=(-1, -1), softcap=0.0, alibi_slopes=None,
                    deterministic=False, return_attn_probs=False):
    """Drop-in replacement for flash_attn.flash_attn_func (B, S, H, D layout)."""
    q_b, k_b, v_b = (t.transpose(1, 2).contiguous() for t in (q, k, v))
    simple = (dropout_p == 0.0 and softmax_scale is None
              and window_size == (-1, -1) and softcap == 0.0
              and alibi_slopes is None)
    if simple and q_b.dtype == torch.float16:
        qs, ks = q_b.shape[2], k_b.shape[2]
        from zluda_ops import flash_decode as _fd
        from zluda_ops import flash_attention as _fa

        # decode: a few query tokens against a long KV cache (GQA supported by
        # the kernel: q_heads % kv_heads == 0)
        if (qs <= 16 and ks >= 512 and ks % 128 == 0
                and q_b.shape[1] % k_b.shape[1] == 0):
            try:
                out = _fd.flash_decode_attn(q_b, k_b, v_b, causal=bool(causal))
                return out.transpose(1, 2).contiguous()
            except NotImplementedError:
                pass
        # prefill: full attention, Triton FA handles GQA internally
        if qs == ks and qs % 128 == 0:
            try:
                out = _fa.flash_attn_func(q_b, k_b, v_b, causal=bool(causal))
                return out.transpose(1, 2).contiguous()
            except NotImplementedError:
                pass
    out = _math_sdpa(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                     dropout_p, softmax_scale, causal)
    return out.transpose(1, 2).contiguous()


def flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k,
                           max_seqlen_q=None, max_seqlen_k=None,
                           dropout_p=0.0, softmax_scale=None, causal=False,
                           window_size=(-1, -1), softcap=0.0,
                           alibi_slopes=None, deterministic=False,
                           return_attn_probs=False, block_table=None):
    """Drop-in replacement for flash_attn.flash_attn_varlen_func."""
    from zluda_ops.flash_attention import flash_attn_varlen_func as _impl

    if q.dim() != 3:
        raise ValueError("varlen expects (total, heads, head_dim) tensors")
    return _impl(q, k, v, cu_seqlens_q, cu_seqlens_k,
                 max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k,
                 dropout_p=dropout_p, softmax_scale=softmax_scale, causal=causal,
                 window_size=window_size, softcap=softcap,
                 alibi_slopes=alibi_slopes, deterministic=deterministic,
                 return_attn_probs=return_attn_probs, block_table=block_table)


__all__ = ["flash_attn_func", "flash_attn_varlen_func", "__version__"]