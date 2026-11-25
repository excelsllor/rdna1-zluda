"""Flash-decoding attention for small query lengths (decode).

During LLM decode the query has 1-8 tokens while the KV cache is long, so the
regular flash-attention kernel (grid over query blocks) runs only B*H programs
and cannot fill the CUs. This module splits the KV cache into chunks, computes
partial (acc, lse) for every chunk in parallel, and merges the partials with
log-sum-exp.

API:
    out = flash_decode_attn(q, k, v, causal=False)

q/k/v are (B, H, M, D) fp16 contiguous, M <= 16, D == 64, and the KV length must
be divisible by BLOCK_N * SPLITS (otherwise NotImplementedError -> caller can
fall back to math SDPA). When `causal` is set, the M query tokens are assumed to
be the last M positions of a length-N sequence.
"""

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _decode_split(Q, K, V, Out, Max, Lsum,
                  stride_qz, stride_qh, stride_qm, stride_qk,
                  stride_kz, stride_kh, stride_kn, stride_kk,
                  stride_vz, stride_vh, stride_vn, stride_vk,
                  H, HKV, GROUP, M, N, OFFSET, sm_scale,
                  SPLITS: tl.constexpr, BLOCK_M: tl.constexpr,
                  BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr,
                  CAUSAL: tl.constexpr):
    pid_split = tl.program_id(0)
    bhid = tl.program_id(1)
    off_z = bhid // H
    off_h = bhid % H
    off_hkv = off_h // GROUP
    q_off = off_z * stride_qz + off_h * stride_qh
    k_off = off_z * stride_kz + off_hkv * stride_kh
    v_off = off_z * stride_vz + off_hkv * stride_vh

    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    q = tl.load(Q + q_off + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk,
                mask=offs_m[:, None] < M, other=0.0)

    chunk = N // SPLITS
    start = pid_split * chunk
    end = start + chunk
    qk_scale = sm_scale * 1.44269504

    m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, HEAD_DIM), tl.float32)
    for start_n in range(start, end, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        k = tl.load(K + k_off + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk)
        v = tl.load(V + v_off + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk)
        qk = tl.dot(q, tl.trans(k)) * qk_scale
        if CAUSAL:
            qpos = OFFSET + offs_m
            qk = tl.where(qpos[:, None] >= offs_n[None, :], qk, -1.0e6)
        m_new = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.math.exp2(qk - m_new[:, None])
        alpha = tl.math.exp2(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16), v)
        m_i = m_new

    part = (bhid * SPLITS + pid_split)
    out_ptrs = Out + part * M * HEAD_DIM + offs_m[:, None] * HEAD_DIM + offs_d[None, :]
    tl.store(out_ptrs, acc, mask=offs_m[:, None] < M)
    # Store the raw (m, l) pair: the merge needs 2^(m_s - m_max) scaling for
    # both the numerator and the denominator (lse alone is not enough).
    tl.store(Max + part * M + offs_m, m_i, mask=offs_m < M)
    tl.store(Lsum + part * M + offs_m, l_i, mask=offs_m < M)


@triton.jit
def _decode_merge(Out, Max, Lsum, Res, M,
                  SPLITS: tl.constexpr, BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr):
    bhid = tl.program_id(0)
    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    base = bhid * SPLITS * M + offs_m
    # m_max = max_s (m_s + log2 l_s): the largest split log-sum-exp.
    m_max = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    for s in tl.static_range(SPLITS):
        ms = tl.load(Max + base + s * M, mask=offs_m < M, other=-float("inf"))
        ls = tl.load(Lsum + base + s * M, mask=offs_m < M, other=0.0)
        lse = ms + tl.math.log2(ls)
        m_max = tl.maximum(m_max, lse)
    acc = tl.zeros((BLOCK_M, HEAD_DIM), tl.float32)
    l_sum = tl.zeros((BLOCK_M,), tl.float32)
    for s in tl.static_range(SPLITS):
        ms = tl.load(Max + base + s * M, mask=offs_m < M, other=-float("inf"))
        ls = tl.load(Lsum + base + s * M, mask=offs_m < M, other=0.0)
        w = tl.math.exp2(ms - m_max)
        w = tl.where(offs_m < M, w, 0.0)
        l_sum += ls * w
        part = tl.load(Out + (bhid * SPLITS + s) * M * HEAD_DIM
                       + offs_m[:, None] * HEAD_DIM + offs_d[None, :],
                       mask=offs_m[:, None] < M, other=0.0)
        acc += part * w[:, None]
    res = acc / l_sum[:, None]
    res_ptrs = Res + bhid * M * HEAD_DIM + offs_m[:, None] * HEAD_DIM + offs_d[None, :]
    tl.store(res_ptrs, res.to(Res.type.element_ty), mask=offs_m[:, None] < M)


BLOCK_N = 128
BLOCK_M = 16


def _choose_splits(n_ctx, batch_heads, target_programs=128):
    """Pick the largest SPLITS <= 16 that divides n_ctx/BLOCK_N and fills the CUs."""
    chunks = n_ctx // BLOCK_N
    best = 1
    for splits in range(1, 17):
        if chunks % splits:
            continue
        if splits * batch_heads >= target_programs:
            return splits
        best = splits
    return best


def flash_decode_attn(q, k, v, causal=False, splits=None):
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise NotImplementedError("flash_decode_attn expects 4D (B,H,M,D) tensors")
    if q.dtype != torch.float16 or k.dtype != torch.float16 or v.dtype != torch.float16:
        raise NotImplementedError("flash_decode_attn supports fp16 only")
    if not (q.is_cuda and k.is_cuda and v.is_cuda):
        raise NotImplementedError("flash_decode_attn needs a CUDA/ZLUDA device")
    b, h, m, d = q.shape
    n = k.shape[2]
    if d != 64 or m > BLOCK_M or k.shape[3] != d or v.shape[3] != d:
        raise NotImplementedError("flash_decode_attn requires head_dim=64 and M<=16")
    if q.shape[0] != k.shape[0] or k.shape[1] != v.shape[1]:
        raise NotImplementedError("q/k/v batch and kv head counts must match")
    if h % k.shape[1]:
        raise NotImplementedError("query heads must be a multiple of kv heads")
    if n % BLOCK_N != 0:
        raise NotImplementedError("KV length must be a multiple of 128")

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    if splits is None:
        splits = _choose_splits(n, b * h)
    if (n // BLOCK_N) % splits:
        splits = 1

    partial = torch.empty((b * h * splits, m, d), device=q.device, dtype=torch.float32)
    maxes = torch.empty((b * h * splits, m), device=q.device, dtype=torch.float32)
    lsums = torch.empty((b * h * splits, m), device=q.device, dtype=torch.float32)
    out = torch.empty_like(q)
    sm_scale = 1.0 / math.sqrt(d)
    offset = n - m
    _decode_split[(splits, b * h)](
        q, k, v, partial, maxes, lsums,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        h, k.shape[1], h // k.shape[1], m, n, offset, sm_scale,
        SPLITS=splits, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=d,
        CAUSAL=causal, num_warps=4, num_stages=1,
    )
    _decode_merge[(b * h,)](
        partial, maxes, lsums, out, m,
        SPLITS=splits, BLOCK_M=BLOCK_M, HEAD_DIM=d,
        num_warps=4, num_stages=1,
    )
    return out


__all__ = ["flash_decode_attn"]
