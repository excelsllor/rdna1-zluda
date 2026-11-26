"""GQA KV-reuse flash-decoding (Phase 2 kernel).

The existing `flash_decode_attn` maps one program to one (batch, query-head)
pair, so with GQA (e.g. 14 q heads / 2 kv heads) every query head of a group
re-reads the same K/V from DRAM: 7x redundant traffic on long contexts.

This kernel maps one program to one **(batch, kv-head)** pair and loops over the
query heads of its group sequentially. The K/V tile for a chunk is loaded once
per iteration; the following head iterations hit the same chunk in L2 (chunk =
N/SPLITS * D * 2 bytes fits in the 4 MB L2), so DRAM traffic drops to ~1x while
keeping SPLITS * B * HKV programs busy across the CUs.

API (mirrors flash_decode):
    out = gqa_decode_attn(q, k, v, causal=False)  # (B, Hq, M, D) fp16 -> (B, Hq, M, D)
Returns None when the shape is not supported (caller falls back).
"""

import torch
import triton
import triton.language as tl

from .flash_decode import BLOCK_M, BLOCK_N, _choose_splits, _decode_merge


@triton.jit
def _gqa_decode_split(Q, K, V, Out, Max, Lsum,
                      stride_qz, stride_qh, stride_qm, stride_qk,
                      stride_kz, stride_kh, stride_kn, stride_kk,
                      stride_vz, stride_vh, stride_vn, stride_vk,
                      Hq, HKV, GROUP, M, N, OFFSET, sm_scale,
                      SPLITS: tl.constexpr, BLOCK_M: tl.constexpr,
                      BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr,
                      CAUSAL: tl.constexpr):
    pid_split = tl.program_id(0)
    bhid_kv = tl.program_id(1)
    off_z = bhid_kv // HKV
    off_hkv = bhid_kv % HKV

    k_off = off_z * stride_kz + off_hkv * stride_kh
    v_off = off_z * stride_vz + off_hkv * stride_vh
    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    qk_scale = sm_scale * 1.44269504

    chunk = N // SPLITS
    start = pid_split * chunk
    end = start + chunk

    for g in range(GROUP):
        off_h = off_hkv * GROUP + g
        q_off = off_z * stride_qz + off_h * stride_qh
        q = tl.load(Q + q_off + offs_m[:, None] * stride_qm
                    + offs_d[None, :] * stride_qk,
                    mask=offs_m[:, None] < M, other=0.0)
        m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
        l_i = tl.zeros((BLOCK_M,), tl.float32)
        acc = tl.zeros((BLOCK_M, HEAD_DIM), tl.float32)
        for start_n in range(start, end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            k = tl.load(K + k_off + offs_n[:, None] * stride_kn
                        + offs_d[None, :] * stride_kk)
            v = tl.load(V + v_off + offs_n[:, None] * stride_vn
                        + offs_d[None, :] * stride_vk)
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

        part = (off_z * Hq + off_h) * SPLITS + pid_split
        out_ptrs = Out + part * M * HEAD_DIM + offs_m[:, None] * HEAD_DIM + offs_d[None, :]
        tl.store(out_ptrs, acc, mask=offs_m[:, None] < M)
        tl.store(Max + part * M + offs_m, m_i, mask=offs_m < M)
        tl.store(Lsum + part * M + offs_m, l_i, mask=offs_m < M)


def gqa_decode_attn(q, k, v, causal=False, splits=None):
    """GQA-aware flash-decoding for (B, Hq, M, D) fp16, M<=16, D=64.

    Returns (B, Hq, M, D) fp16 or None if the shape is unsupported.
    """
    if (not q.is_cuda or q.dim() != 4 or q.dtype != torch.float16
            or k.dtype != torch.float16 or v.dtype != torch.float16):
        return None
    b, hq, m, d = q.shape
    if m < 1 or m > BLOCK_M or d != 64:
        return None
    if k.shape[0] != b or v.shape[0] != b or k.shape[2] != v.shape[2]:
        return None
    if k.shape[1] != v.shape[1] or k.shape[3] != d or v.shape[3] != d:
        return None
    hkv = k.shape[1]
    if hkv < 1 or hq % hkv:
        return None
    group = hq // hkv
    n = k.shape[2]
    if n < 512 or n % (BLOCK_N * 2):
        return None
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    if splits is None:
        splits = _choose_splits(n, b * hq)
    if n % (BLOCK_N * splits):
        return None
    partials = torch.empty(b * hq * splits * m * d, device=q.device,
                           dtype=torch.float32)
    maxs = torch.empty(b * hq * splits * m, device=q.device, dtype=torch.float32)
    lsums = torch.empty_like(maxs)
    out = torch.empty(b, hq, m, d, device=q.device, dtype=torch.float16)
    sm_scale = (1.0 / (d ** 0.5))
    offset = n - m
    _gqa_decode_split[(splits, b * hkv)](
        q, k, v, partials, maxs, lsums,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        hq, hkv, group, m, n, offset, sm_scale,
        SPLITS=splits, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=d,
        CAUSAL=causal, num_warps=4, num_stages=1,
    )
    _decode_merge[(b * hq,)](partials, maxs, lsums, out, m,
                             SPLITS=splits, BLOCK_M=BLOCK_M, HEAD_DIM=d,
                             num_warps=4)
    return out


__all__ = ["gqa_decode_attn"]
