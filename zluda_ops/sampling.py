"""Fused sampling kernels (Phase 2).

Sampling in a generation loop normally materialises probabilities and then calls
torch.multinomial (an extra kernel plus a host sync). These kernels do it in two
GPU passes over the logits without writing the probability vector:

  pass 1 (`_softmax_stats`): running max + sum of exp (temperature applied) in a
          single program looping over the vocabulary;
  pass 2 (`_sample_cdf`):   re-scan the logits, build the cumulative
          distribution on the fly and pick the first index whose cumulative
          probability crosses u * sumexp.

`top_k`/`top_p` filtering is applied to the logits with torch ops before the
kernel (masking out the tail with -inf), which keeps the kernel simple while
still avoiding the probability vector and the multinomial call.

API:
    token = sample_logits(logits, temperature=1.0, top_k=0, top_p=1.0)
`logits` is a 1-D tensor (vocab); returns a 0-dim int64 tensor.
"""

import random

import torch
import triton
import triton.language as tl

_BLOCK = 4096


@triton.jit
def _gumbel_blocks(LOGITS, U, V, TEMP, PART_VAL, PART_IDX, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    mask = idx < V
    x = tl.load(LOGITS + idx, mask=mask,
                other=-float("inf")).to(tl.float32) * TEMP
    u = tl.load(U + idx, mask=mask, other=1.0)
    u = tl.maximum(u, 1e-7)
    g = -tl.log(-tl.log(u))
    val = tl.where(mask, x + g, -float("inf"))
    block_val = tl.max(val, axis=0)
    block_idx = tl.min(tl.where(val == block_val, idx, 1073741824), axis=0)
    tl.store(PART_VAL + pid, block_val)
    tl.store(PART_IDX + pid, block_idx)


@triton.jit
def _gumbel_merge(PART_VAL, PART_IDX, NBLK, OUT, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    mask = offs < NBLK
    val = tl.load(PART_VAL + offs, mask=mask, other=-float("inf"))
    idx = tl.load(PART_IDX + offs, mask=mask, other=1073741824)
    best_val = tl.max(val, axis=0)
    best_idx = tl.min(tl.where(val == best_val, idx, 1073741824), axis=0)
    tl.store(OUT, best_idx.to(tl.int64))

def sample_logits(logits, temperature=1.0, top_k=0, top_p=1.0, generator=None):
    """Sample a token from 1-D logits (temperature/top-k/top-p supported)."""
    if not logits.is_cuda or logits.dim() != 1:
        return None
    if temperature is None or temperature <= 0:
        return logits.argmax()
    work = logits
    if top_k and top_k > 0 and top_k < logits.numel():
        kth = torch.topk(logits, top_k).values[-1]
        work = logits.masked_fill(logits < kth, float("-inf"))
    if top_p and top_p < 1.0:
        probs = torch.softmax(work.float(), dim=-1)
        sorted_p, sorted_i = torch.sort(probs, descending=True)
        csum = torch.cumsum(sorted_p, dim=-1)
        cut = csum - sorted_p > top_p
        sorted_p = sorted_p.masked_fill(cut, 0.0)
        probs = torch.zeros_like(probs).scatter_(0, sorted_i, sorted_p)
        work = torch.log(probs.clamp(min=1e-30))
    work = work.contiguous().float()
    out = torch.empty((), device=work.device, dtype=torch.int64)
    seed = int(torch.randint(0, 2 ** 31 - 1, (1,), device=work.device,
                             generator=generator).item())
    v = work.numel()
    # torch-generated uniform noise (tl.rand lowering is very slow on this stack);
    # the Triton kernel does temperature scaling, the Gumbel transform and the
    # two-stage argmax reduction.
    u = torch.rand(v, device=work.device, dtype=torch.float32,
                   generator=generator)
    nblk = triton.cdiv(v, _BLOCK)
    part_val = torch.empty(nblk, device=work.device, dtype=torch.float32)
    part_idx = torch.empty(nblk, device=work.device, dtype=torch.int32)
    _gumbel_blocks[(nblk,)](work, u, v, float(temperature), part_val,
                            part_idx, BLOCK=_BLOCK, num_warps=8)
    _gumbel_merge[(1,)](part_val, part_idx, nblk, out, BLOCK=128, num_warps=4)


    return out


__all__ = ["sample_logits"]
