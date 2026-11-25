"""flash_attn.bert_padding-compatible helpers (eager PyTorch implementations).

These are only used by code that unpads padded batches before calling
flash_attn_varlen_func; the implementations mirror the upstream flash-attn
semantics without any CUDA kernels.
"""

import torch


def index_first_axis(x, indices):
    return x.index_select(0, indices)


def unpad_input(hidden_states, attention_mask):
    """(B, S, ...) + (B, S) mask -> (total, ...), indices, cu_seqlens, max_len."""
    if attention_mask.dim() == 4:  # HF sometimes passes (B, 1, S, S)
        attention_mask = attention_mask[:, 0, 0, :]
    seqlens = attention_mask.sum(dim=-1, dtype=torch.int32)
    flat_mask = attention_mask.flatten()
    indices = torch.nonzero(flat_mask, as_tuple=False).flatten()
    max_seqlen = int(seqlens.max().item()) if seqlens.numel() else 0
    cu_seqlens = torch.zeros(seqlens.shape[0] + 1, dtype=torch.int32,
                             device=attention_mask.device)
    cu_seqlens[1:] = torch.cumsum(seqlens, dim=0)
    flat = hidden_states.reshape(-1, *hidden_states.shape[2:])
    return index_first_axis(flat, indices), indices, cu_seqlens, max_seqlen


def pad_input(hidden_states, indices, batch, seqlen):
    """(total, ...) + indices -> (B, S, ...)."""
    out = hidden_states.new_zeros(batch * seqlen, *hidden_states.shape[1:])
    out.index_copy_(0, indices, hidden_states)
    return out.reshape(batch, seqlen, *hidden_states.shape[1:])


__all__ = ["unpad_input", "pad_input", "index_first_axis"]
