"""Fused embedding gather (+ scale), Phase 2.

y[i] = weight[idx[i]] * scale in one kernel (one program per token), instead of
index_select + multiply. Weight rows are contiguous; the kernel is memory-bound
and is usually launched as part of the embedding lookup of a model.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _embedding_kernel(IDX, W, Y, SCALE, D: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    tok = tl.load(IDX + pid)
    offs = tl.arange(0, BLOCK)
    mask = offs < D
    w = tl.load(W + tok * D + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(Y + pid * D + offs, (w * SCALE).to(Y.dtype.element_ty), mask=mask)


def embedding_lookup(idx, weight, scale=1.0):
    """Fused gather+scale for 1-D int64/32 indices; returns (N, D) fp16."""
    if (not weight.is_cuda or weight.dtype != torch.float16 or weight.dim() != 2
            or not idx.is_cuda or idx.dim() != 1):
        return None
    d = weight.shape[1]
    if d > 8192:
        return None
    block = triton.next_power_of_2(d)
    n = idx.shape[0]
    y = torch.empty(n, d, device=weight.device, dtype=torch.float16)
    if n == 0:
        return y
    _embedding_kernel[(n,)](idx.contiguous(), weight, y, float(scale),
                            D=d, BLOCK=block, num_warps=4, num_stages=1)
    return y


__all__ = ["embedding_lookup"]