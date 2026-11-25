"""flash_attn-compatible shim backed by the ZLUDA Triton kernels.

`pip install flash-attn` cannot work on this stack: the package needs nvcc, and
its CUDA kernels target sm80+ SASS/PTX while RDNA1 has no matrix cores. This
module provides the public surface that inference/training code and
`transformers` expect, implemented with `zluda_ops.flash_attention` (Triton FA
forward+backward, math fallback for unsupported shapes) and
`zluda_ops.flash_decode` (split-KV decode).

Install with `tools/install_ecosystem.py` (adds `integrations/` and the repo root
to sys.path) or set `PYTHONPATH=integrations;.`.

Limitations: no real CUDA flash-attention kernels; the Triton path needs
head_dim=64 and fp16; everything else falls back to PyTorch math SDPA (correct,
slower). `flash_attn_func` expects (B, S, H, D) like the real package.
"""

import torch
import torch.nn.functional as F

__version__ = "2.7.4+zluda-shim"


def _math_sdpa(q, k, v, dropout_p, scale, causal):
    from torch.nn.attention import SDPBackend, sdpa_kernel

    with sdpa_kernel(SDPBackend.MATH):
        return F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=dropout_p, is_causal=causal,
            scale=scale,
        )


def flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False,
                    window_size=(-1, -1), softcap=0.0, alibi_slopes=None,
                    deterministic=False, return_attn_probs=False):
    """Drop-in replacement for flash_attn.flash_attn_func (B, S, H, D layout)."""
    q_bhsd, k_bhsd, v_bhsd = (t.transpose(1, 2).contiguous() for t in (q, k, v))
    try:
        from zluda_ops.flash_attention import flash_attn_func as _impl

        out = _impl(q_bhsd, k_bhsd, v_bhsd, dropout_p=dropout_p,
                    softmax_scale=softmax_scale, causal=causal,
                    window_size=window_size, softcap=softcap,
                    alibi_slopes=alibi_slopes, deterministic=deterministic,
                    return_attn_probs=return_attn_probs)
        return out.transpose(1, 2).contiguous()
    except NotImplementedError:
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


class FlashAttnFunc(torch.autograd.Function):
    """Minimal autograd wrapper for code that instantiates the class."""

    @staticmethod
    def forward(ctx, q, k, v, dropout_p=0.0, softmax_scale=None, causal=False,
                window_size=(-1, -1), softcap=0.0, alibi_slopes=None,
                deterministic=False, return_attn_probs=False):
        ctx.save_for_backward(q, k, v)
        ctx.params = (dropout_p, softmax_scale, causal, window_size, softcap,
                      deterministic)
        return flash_attn_func(q, k, v, dropout_p=dropout_p,
                               softmax_scale=softmax_scale, causal=causal,
                               window_size=window_size, softcap=softcap,
                               deterministic=deterministic)

    @staticmethod
    def backward(ctx, grad_out):
        q, k, v = ctx.saved_tensors
        dropout_p, softmax_scale, causal, window_size, softcap, _ = ctx.params
        with torch.enable_grad():
            q_ = q.detach().requires_grad_(True)
            k_ = k.detach().requires_grad_(True)
            v_ = v.detach().requires_grad_(True)
            out = flash_attn_func(q_, k_, v_, dropout_p=dropout_p,
                                  softmax_scale=softmax_scale, causal=causal,
                                  window_size=window_size, softcap=softcap)
            gq, gk, gv = torch.autograd.grad(out, (q_, k_, v_), grad_out)
        return gq, gk, gv, None, None, None, None, None, None, None, None

__all__ = ["flash_attn_func", "flash_attn_varlen_func", "FlashAttnFunc",
           "__version__"]
