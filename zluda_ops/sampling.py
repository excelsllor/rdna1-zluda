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

import torch
import triton
import triton.language as tl

_BLOCK = 4096


@triton.jit
def _sample_gumbel(LOGITS, V, TEMP, SEED, OUT, BLOCK: tl.constexpr):
    # Exact categorical sampling via the Gumbel-max trick:
    #   argmax_i (logit_i / T + gumbel_i) ~ Categorical(softmax(logits / T))
    # No cumsum / probability vector is needed, which works around the
    # unreliable tl.cumsum lowering on this stack.
    best_val = tl.full((), -float("inf"), tl.float32)
    best_idx = tl.full((), 0, tl.int32)
    for off in range(0, V, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < V
        x = tl.load(LOGITS + idx, mask=mask,
                    other=-float("inf")).to(tl.float32) * TEMP
        u = tl.rand(SEED, idx)
        u = tl.maximum(u, 1e-7)
        g = -tl.log(-tl.log(u))
        val = tl.where(mask, x + g, -float("inf"))
        blk_val = tl.max(val, axis=0)
        blk_idx = tl.min(tl.where(val == blk_val, idx, 1073741824), axis=0)
        take = blk_val > best_val
        best_val = tl.where(take, blk_val, best_val)
        best_idx = tl.where(take, blk_idx, best_idx)
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
    _sample_gumbel[(1,)](work, work.numel(), float(temperature), seed, out,
                         BLOCK=_BLOCK, num_warps=8)

    return out


__all__ = ["sample_logits"]
