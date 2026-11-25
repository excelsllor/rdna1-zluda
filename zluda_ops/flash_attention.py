import math

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel
@triton.jit
def _attn_fwd_inner(acc, l_i, m_i, q,
                    K_block_ptr, V_block_ptr,
                    start_m, qk_scale,
                    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,
                    STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,
                    N_CTX: tl.constexpr,
                    Mask, mask_stride_m, mask_stride_n,
                    MASK_TYPE: tl.constexpr):
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
    else:
        lo, hi = 0, N_CTX
    K_block_ptr = tl.advance(K_block_ptr, (0, lo))
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k = tl.load(K_block_ptr)
        qk = tl.dot(q, k)
        qk = qk * qk_scale
        # Attn mask first (bool: drop; additive: add in log2 domain), then the
        # causal constraint so a positive additive bias cannot resurrect a
        # causally masked position.
        if MASK_TYPE == 1:
            mvals = tl.load(Mask + offs_m[:, None] * mask_stride_m
                            + (start_n + offs_n[None, :]) * mask_stride_n)
            qk = tl.where(mvals != 0, qk, -1.0e6)
        elif MASK_TYPE == 2:
            mvals = tl.load(Mask + offs_m[:, None] * mask_stride_m
                            + (start_n + offs_n[None, :]) * mask_stride_n)
            qk += mvals.to(tl.float32) * 1.4426950408889634
        if STAGE == 2:
            cmask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = tl.where(cmask, qk, -1.0e6)
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk -= m_ij[:, None]
        p = tl.math.exp2(qk)
        if MASK_TYPE == 1:
            # -1e6 survives the row-max subtraction; zero masked lanes so a
            # partially masked row does not average over masked values.
            p = tl.where(mvals != 0, p, 0.0)
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]
        v = tl.load(V_block_ptr)
        p = p.to(tl.float16)
        acc = tl.dot(p, v, acc)
        m_i = m_ij
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
    return acc, l_i, m_i


@triton.jit
def _attn_fwd(Q, K, V, sm_scale, M, Out,
              stride_qz, stride_qh, stride_qm, stride_qk,
              stride_kz, stride_kh, stride_kn, stride_kk,
              stride_vz, stride_vh, stride_vk, stride_vn,
              stride_oz, stride_oh, stride_om, stride_on,
              Z, H, N_CTX,
              Mask, mask_stride_z, mask_stride_h, mask_stride_m, mask_stride_n,
              HEAD_DIM: tl.constexpr,
              BLOCK_M: tl.constexpr,
              BLOCK_N: tl.constexpr,
              STAGE: tl.constexpr,
              MASK_TYPE: tl.constexpr):
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    qvk_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
    Mask += off_z.to(tl.int64) * mask_stride_z + off_h.to(tl.int64) * mask_stride_h

    Q_block_ptr = tl.make_block_ptr(
        base=Q + qvk_offset, shape=(N_CTX, HEAD_DIM), strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0), block_shape=(BLOCK_M, HEAD_DIM), order=(1, 0))
    V_block_ptr = tl.make_block_ptr(
        base=V + qvk_offset, shape=(N_CTX, HEAD_DIM), strides=(stride_vk, stride_vn),
        offsets=(0, 0), block_shape=(BLOCK_N, HEAD_DIM), order=(1, 0))
    K_block_ptr = tl.make_block_ptr(
        base=K + qvk_offset, shape=(HEAD_DIM, N_CTX), strides=(stride_kk, stride_kn),
        offsets=(0, 0), block_shape=(HEAD_DIM, BLOCK_N), order=(0, 1))
    O_block_ptr = tl.make_block_ptr(
        base=Out + qvk_offset, shape=(N_CTX, HEAD_DIM), strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0), block_shape=(BLOCK_M, HEAD_DIM), order=(1, 0))

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    qk_scale = sm_scale * 1.44269504
    q = tl.load(Q_block_ptr)
    if STAGE & 1:
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, K_block_ptr, V_block_ptr,
                                        start_m, qk_scale, BLOCK_M, HEAD_DIM, BLOCK_N,
                                        4 - STAGE, offs_m, offs_n, N_CTX,
                                        Mask, mask_stride_m, mask_stride_n, MASK_TYPE)
    if STAGE & 2:
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, K_block_ptr, V_block_ptr,
                                        start_m, qk_scale, BLOCK_M, HEAD_DIM, BLOCK_N,
                                        2, offs_m, offs_n, N_CTX,
                                        Mask, mask_stride_m, mask_stride_n, MASK_TYPE)
    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    m_ptrs = M + off_hz * N_CTX + offs_m
    tl.store(m_ptrs, m_i)
    tl.store(O_block_ptr, acc.to(Out.type.element_ty))


@triton.jit
def _attn_bwd_preprocess(O, DO, Delta, Z, H, N_CTX,
                         BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr):
    off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    off_hz = tl.program_id(1)
    off_n = tl.arange(0, HEAD_DIM)
    o = tl.load(O + off_hz * HEAD_DIM * N_CTX + off_m[:, None] * HEAD_DIM + off_n[None, :])
    do = tl.load(DO + off_hz * HEAD_DIM * N_CTX + off_m[:, None] * HEAD_DIM + off_n[None, :]).to(tl.float32)
    delta = tl.sum(o * do, axis=1)
    tl.store(Delta + off_hz * N_CTX + off_m, delta)


@triton.jit
def _attn_bwd_dkdv(dk, dv, Q, k, v, sm_scale, DO, M, D,
                   stride_tok, stride_d, H, N_CTX,
                   BLOCK_M1: tl.constexpr, BLOCK_N1: tl.constexpr, HEAD_DIM: tl.constexpr,
                   start_n, start_m, num_steps,
                   Mask, mask_stride_m, mask_stride_n,
                   MASK_TYPE: tl.constexpr, MASK: tl.constexpr):
    offs_m = start_m + tl.arange(0, BLOCK_M1)
    offs_n = start_n + tl.arange(0, BLOCK_N1)
    offs_k = tl.arange(0, HEAD_DIM)
    qT_ptrs = Q + offs_m[None, :] * stride_tok + offs_k[:, None] * stride_d
    do_ptrs = DO + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d
    tl.static_assert(BLOCK_N1 % BLOCK_M1 == 0)
    curr_m = start_m
    step_m = BLOCK_M1
    for blk_idx in range(num_steps):
        qT = tl.load(qT_ptrs)
        offs_m = curr_m + tl.arange(0, BLOCK_M1)
        m = tl.load(M + offs_m)
        qkT = tl.dot(k, qT)
        if MASK_TYPE == 2:
            mv = tl.load(Mask + offs_m[None, :] * mask_stride_m
                         + offs_n[:, None] * mask_stride_n)
            qkT += mv.to(tl.float32) * 1.4426950408889634
        pT = tl.math.exp2(qkT - m[None, :])
        if MASK_TYPE == 1:
            mv = tl.load(Mask + offs_m[None, :] * mask_stride_m
                         + offs_n[:, None] * mask_stride_n)
            pT = tl.where(mv != 0, pT, 0.0)
        if MASK:
            mask = (offs_m[None, :] >= offs_n[:, None])
            pT = tl.where(mask, pT, 0.0)
        do = tl.load(do_ptrs)
        ppT = pT.to(tl.float16)
        dv += tl.dot(ppT, do)
        Di = tl.load(D + offs_m)
        dpT = tl.dot(v, tl.trans(do)).to(tl.float32)
        dsT = pT * (dpT - Di[None, :])
        dsT = dsT.to(tl.float16)
        dk += tl.dot(dsT, tl.trans(qT))
        curr_m += step_m
        qT_ptrs += step_m * stride_tok
        do_ptrs += step_m * stride_tok
    return dk, dv


@triton.jit
def _attn_bwd_dq(dq, q, K, V, do, m, D,
                 stride_tok, stride_d, H, N_CTX,
                 BLOCK_M2: tl.constexpr, BLOCK_N2: tl.constexpr, HEAD_DIM: tl.constexpr,
                 start_m, start_n, num_steps,
                 Mask, mask_stride_m, mask_stride_n,
                 MASK_TYPE: tl.constexpr, MASK: tl.constexpr):
    offs_m = start_m + tl.arange(0, BLOCK_M2)
    offs_n = start_n + tl.arange(0, BLOCK_N2)
    offs_k = tl.arange(0, HEAD_DIM)
    kT_ptrs = K + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d
    vT_ptrs = V + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d
    Di = tl.load(D + offs_m)
    tl.static_assert(BLOCK_M2 % BLOCK_N2 == 0)
    curr_n = start_n
    step_n = BLOCK_N2
    for blk_idx in range(num_steps):
        kT = tl.load(kT_ptrs)
        vT = tl.load(vT_ptrs)
        qk = tl.dot(q, kT)
        offs_n_blk = curr_n + tl.arange(0, BLOCK_N2)
        qk_adj = qk
        if MASK_TYPE == 2:
            mv = tl.load(Mask + offs_m[:, None] * mask_stride_m
                         + offs_n_blk[None, :] * mask_stride_n)
            qk_adj = qk + mv.to(tl.float32) * 1.4426950408889634
        p = tl.math.exp2(qk_adj - m)
        if MASK_TYPE == 1:
            mv = tl.load(Mask + offs_m[:, None] * mask_stride_m
                         + offs_n_blk[None, :] * mask_stride_n)
            p = tl.where(mv != 0, p, 0.0)
        if MASK:
            mask = (offs_m[:, None] >= offs_n_blk[None, :])
            p = tl.where(mask, p, 0.0)
        dp = tl.dot(do, vT).to(tl.float32)
        ds = p * (dp - Di[:, None])
        ds = ds.to(tl.float16)
        dq += tl.dot(ds, tl.trans(kT))
        curr_n += step_n
        kT_ptrs += step_n * stride_tok
        vT_ptrs += step_n * stride_tok
    return dq


@triton.jit
def _attn_bwd(Q, K, V, sm_scale, DO, DQ, DK, DV, M, D,
              stride_z, stride_h, stride_tok, stride_d, H, N_CTX,
              Mask, mask_stride_z, mask_stride_h, mask_stride_m, mask_stride_n,
              BLOCK_M1: tl.constexpr, BLOCK_N1: tl.constexpr,
              BLOCK_M2: tl.constexpr, BLOCK_N2: tl.constexpr,
              BLK_SLICE_FACTOR: tl.constexpr, HEAD_DIM: tl.constexpr,
              CAUSAL: tl.constexpr, MASK_TYPE: tl.constexpr):
    LN2: tl.constexpr = 0.6931471824645996
    bhid = tl.program_id(2)
    off_chz = (bhid * N_CTX).to(tl.int64)
    adj = (stride_h * (bhid % H) + stride_z * (bhid // H)).to(tl.int64)
    pid = tl.program_id(0)

    Q += adj
    K += adj
    V += adj
    DO += adj
    DQ += adj
    DK += adj
    DV += adj
    M += off_chz
    D += off_chz
    Mask += (bhid // H) * mask_stride_z + (bhid % H) * mask_stride_h

    offs_k = tl.arange(0, HEAD_DIM)
    start_n = pid * BLOCK_N1
    MASK_BLOCK_M1: tl.constexpr = BLOCK_M1 // BLK_SLICE_FACTOR
    offs_n = start_n + tl.arange(0, BLOCK_N1)
    dv = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)
    k = tl.load(K + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d)
    v = tl.load(V + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d)
    if CAUSAL:
        start_m = start_n
        num_steps = BLOCK_N1 // MASK_BLOCK_M1
        dk, dv = _attn_bwd_dkdv(dk, dv, Q, k, v, sm_scale, DO, M, D,
                                stride_tok, stride_d, H, N_CTX,
                                MASK_BLOCK_M1, BLOCK_N1, HEAD_DIM,
                                start_n, start_m, num_steps, Mask, mask_stride_m, mask_stride_n, MASK_TYPE, MASK=True)
        start_m += num_steps * MASK_BLOCK_M1
        num_steps = (N_CTX - start_m) // BLOCK_M1
        dk, dv = _attn_bwd_dkdv(dk, dv, Q, k, v, sm_scale, DO, M, D,
                                stride_tok, stride_d, H, N_CTX,
                                BLOCK_M1, BLOCK_N1, HEAD_DIM,
                                start_n, start_m, num_steps, Mask, mask_stride_m, mask_stride_n, MASK_TYPE, MASK=False)
    else:
        num_steps = N_CTX // BLOCK_M1
        dk, dv = _attn_bwd_dkdv(dk, dv, Q, k, v, sm_scale, DO, M, D,
                                stride_tok, stride_d, H, N_CTX,
                                BLOCK_M1, BLOCK_N1, HEAD_DIM,
                                start_n, 0, num_steps, Mask, mask_stride_m, mask_stride_n, MASK_TYPE, MASK=False)
    dv_ptrs = DV + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d
    tl.store(dv_ptrs, dv)
    dk *= sm_scale
    dk_ptrs = DK + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d
    tl.store(dk_ptrs, dk)

    start_m = pid * BLOCK_M2
    MASK_BLOCK_N2: tl.constexpr = BLOCK_N2 // BLK_SLICE_FACTOR
    offs_m = start_m + tl.arange(0, BLOCK_M2)
    q = tl.load(Q + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d)
    dq = tl.zeros([BLOCK_M2, HEAD_DIM], dtype=tl.float32)
    do = tl.load(DO + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d)
    m = tl.load(M + offs_m)
    m = m[:, None]
    if CAUSAL:
        end_n = start_m + BLOCK_M2
        num_steps = BLOCK_M2 // MASK_BLOCK_N2
        dq = _attn_bwd_dq(dq, q, K, V, do, m, D, stride_tok, stride_d, H, N_CTX,
                          BLOCK_M2, MASK_BLOCK_N2, HEAD_DIM,
                          start_m, end_n - num_steps * MASK_BLOCK_N2, num_steps, Mask, mask_stride_m, mask_stride_n, MASK_TYPE, MASK=True)
        end_n -= num_steps * MASK_BLOCK_N2
        num_steps = end_n // BLOCK_N2
        dq = _attn_bwd_dq(dq, q, K, V, do, m, D, stride_tok, stride_d, H, N_CTX,
                          BLOCK_M2, BLOCK_N2, HEAD_DIM,
                          start_m, end_n - num_steps * BLOCK_N2, num_steps, Mask, mask_stride_m, mask_stride_n, MASK_TYPE, MASK=False)
    else:
        num_steps = N_CTX // BLOCK_N2
        dq = _attn_bwd_dq(dq, q, K, V, do, m, D, stride_tok, stride_d, H, N_CTX,
                          BLOCK_M2, BLOCK_N2, HEAD_DIM,
                          start_m, 0, num_steps, Mask, mask_stride_m, mask_stride_n, MASK_TYPE, MASK=False)
    dq_ptrs = DQ + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d
    dq *= LN2
    tl.store(dq_ptrs, dq)


class TritonFlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal, sm_scale, attn_mask, mask_type, mask_strides):
        head_dim = q.shape[-1]
        o = torch.empty_like(q)
        stage = 3 if causal else 1
        lse = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
        grid = (triton.cdiv(q.shape[2], 64), q.shape[0] * q.shape[1], 1)
        mz, mh, mm, mn = mask_strides
        mask_arg = attn_mask if attn_mask is not None else q
        _attn_fwd[grid](
            q, k, v, sm_scale, lse, o,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            q.shape[0], q.shape[1], q.shape[2],
            mask_arg, mz, mh, mm, mn,
            HEAD_DIM=head_dim, BLOCK_M=64, BLOCK_N=64, STAGE=stage,
            MASK_TYPE=mask_type,
            num_warps=4, num_stages=1,
        )
        ctx.save_for_backward(q, k, v, o, lse)
        ctx.sm_scale = sm_scale
        ctx.causal = causal
        ctx.attn_mask = attn_mask
        ctx.mask_type = mask_type
        ctx.mask_strides = mask_strides
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, lse = ctx.saved_tensors
        batch, n_head, n_ctx = q.shape[:3]
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        pre_block = 128
        delta = torch.empty_like(lse)
        _attn_bwd_preprocess[(n_ctx // pre_block, batch * n_head)](
            o, do, delta, batch, n_head, n_ctx,
            BLOCK_M=pre_block, HEAD_DIM=q.shape[-1],
            num_warps=4, num_stages=1,
        )
        rcp_ln2 = 1.4426950408889634
        arg_k = k * (ctx.sm_scale * rcp_ln2)
        block_m1, block_n1, block_m2, block_n2 = 32, 128, 128, 32
        grid = (n_ctx // block_n1, 1, batch * n_head)
        mz, mh, mm, mn = ctx.mask_strides
        mask_arg = ctx.attn_mask if ctx.attn_mask is not None else q
        _attn_bwd[grid](
            q, arg_k, v, ctx.sm_scale, do, dq, dk, dv,
            lse, delta,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            n_head, n_ctx,
            mask_arg, mz, mh, mm, mn,
            BLOCK_M1=block_m1, BLOCK_N1=block_n1,
            BLOCK_M2=block_m2, BLOCK_N2=block_n2,
            BLK_SLICE_FACTOR=2, HEAD_DIM=q.shape[-1], CAUSAL=ctx.causal,
            MASK_TYPE=ctx.mask_type,
            num_warps=8, num_stages=1,
        )
        return dq, dk, dv, None, None, None, None, None


def _unsupported(q, k, v, attn_mask, dropout_p):
    return (
        dropout_p != 0.0
        or q.dtype != torch.float16
        or k.dtype != torch.float16
        or v.dtype != torch.float16
        or not q.is_cuda
        or q.shape[-1] != 64
        or q.shape[2] != k.shape[2]
        or q.shape[2] != v.shape[2]
        or q.shape[2] % 128 != 0
    )


def _prepare_attn_mask(attn_mask, q, k):
    """Normalize an SDPA attention mask for the Triton kernel.

    Returns ``(mask, mask_type, strides)`` where ``mask_type`` is 0 (no mask),
    1 (boolean, stored as uint8) or 2 (additive float32). The mask is broadcast
    to ``(B, H, Sq, Sk)``; broadcast axes keep stride 0 and the kernel handles
    them, so no dense mask is materialised for head/batch-broadcast masks.
    """
    if attn_mask is None:
        return None, 0, (0, 0, 0, 0)
    if attn_mask.dim() == 2:
        mask = attn_mask[None, None]
    elif attn_mask.dim() == 3:
        mask = attn_mask[:, None]
    elif attn_mask.dim() == 4:
        mask = attn_mask
    else:
        raise ValueError("attn_mask must be 2D, 3D or 4D")
    if mask.dtype == torch.bool:
        mask = mask.to(torch.uint8)
        mask_type = 1
    elif mask.is_floating_point():
        mask = mask.to(torch.float32)
        mask_type = 2
    else:
        raise ValueError("attn_mask must be bool or floating point")
    batch, heads = q.shape[0], q.shape[1]
    sq, sk = q.shape[2], k.shape[2]
    try:
        mask = mask.expand(batch, heads, sq, sk)
    except RuntimeError as exc:
        raise ValueError(
            f"attn_mask of shape {tuple(mask.shape)} is not broadcastable to "
            f"{(batch, heads, sq, sk)}"
        ) from exc
    return mask, mask_type, (
        mask.stride(0), mask.stride(1), mask.stride(2), mask.stride(3)
    )


def _math_sdpa(q, k, v, attn_mask, dropout_p, is_causal, scale, enable_gqa):
    if k.shape[1] != q.shape[1] and enable_gqa:
        groups = q.shape[1] // k.shape[1]
        k = k.repeat_interleave(groups, dim=1)
        v = v.repeat_interleave(groups, dim=1)
        enable_gqa = False
    with sdpa_kernel(SDPBackend.MATH):
        return F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=dropout_p,
            is_causal=is_causal, scale=scale, enable_gqa=enable_gqa,
        )


def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0,
                                 is_causal=False, scale=None, enable_gqa=False):
    q, k, v = query, key, value
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])
    if _unsupported(q, k, v, attn_mask, dropout_p):
        return _math_sdpa(q, k, v, attn_mask, dropout_p, is_causal, scale, enable_gqa)
    try:
        mask, mask_type, mask_strides = _prepare_attn_mask(attn_mask, q, k)
    except (ValueError, TypeError):
        return _math_sdpa(q, k, v, attn_mask, dropout_p, is_causal, scale, enable_gqa)
    if k.shape[1] != q.shape[1]:
        if q.shape[1] % k.shape[1] != 0:
            raise ValueError("query heads must be a multiple of key/value heads")
        groups = q.shape[1] // k.shape[1]
        k = k.repeat_interleave(groups, dim=1)
        v = v.repeat_interleave(groups, dim=1)
    return TritonFlashAttention.apply(
        q, k, v, bool(is_causal), float(scale), mask, mask_type, mask_strides,
    )


def _window_mask(sq, sk, causal, window_size, device):
    qi = torch.arange(sq, device=device)[:, None]
    ki = torch.arange(sk, device=device)[None, :]
    allowed = torch.ones((sq, sk), dtype=torch.bool, device=device)
    if causal:
        allowed &= qi >= ki
    left, right = window_size
    if left >= 0:
        allowed &= (qi - ki) <= left
    if right >= 0:
        allowed &= (ki - qi) <= right
    return ~allowed


def flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False,
                    window_size=(-1, -1), softcap=0.0, alibi_slopes=None,
                    deterministic=False, return_attn_probs=False):
    if softcap != 0.0 or alibi_slopes is not None or return_attn_probs:
        raise NotImplementedError("softcap, alibi and attention probabilities are not implemented")
    if window_size != (-1, -1):
        mask = _window_mask(q.shape[2], k.shape[2], causal, window_size, q.device)
        return scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=dropout_p, is_causal=False, scale=softmax_scale,
        )
    return scaled_dot_product_attention(
        q, k, v, attn_mask=None, dropout_p=dropout_p, is_causal=causal, scale=softmax_scale,
    )


def flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q=None,
                           max_seqlen_k=None, dropout_p=0.0, softmax_scale=None,
                           causal=False, window_size=(-1, -1), softcap=0.0,
                           alibi_slopes=None, deterministic=False,
                           return_attn_probs=False, block_table=None):
    if softcap != 0.0 or alibi_slopes is not None or return_attn_probs or block_table is not None:
        raise NotImplementedError("softcap, alibi, paged attention and attention probabilities are not implemented")
    if q.dim() != 3 or k.dim() != 3 or v.dim() != 3:
        raise ValueError("varlen expects q/k/v shaped [total_tokens, heads, head_dim]")
    qs = cu_seqlens_q.tolist()
    ks = cu_seqlens_k.tolist()
    if len(qs) != len(ks):
        raise ValueError("cu_seqlens_q and cu_seqlens_k must have the same length")
    outs = []
    for i in range(len(qs) - 1):
        qa, qb = qs[i], qs[i + 1]
        ka, kb = ks[i], ks[i + 1]
        qi = q[qa:qb].transpose(0, 1).unsqueeze(0)
        ki = k[ka:kb].transpose(0, 1).unsqueeze(0)
        vi = v[ka:kb].transpose(0, 1).unsqueeze(0)
        mask = None
        if window_size != (-1, -1):
            mask = _window_mask(qi.shape[2], ki.shape[2], causal, window_size, q.device)
        oi = scaled_dot_product_attention(
            qi, ki, vi, attn_mask=mask, dropout_p=dropout_p,
            is_causal=(causal and mask is None), scale=softmax_scale,
        )
        outs.append(oi[0].transpose(0, 1))
    return torch.cat(outs, dim=0)


class FlashAttention(nn.Module):
    def __init__(self, causal=False, softmax_scale=None, dropout_p=0.0):
        super().__init__()
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.dropout_p = dropout_p

    def forward(self, q, k, v):
        return scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=self.dropout_p,
            is_causal=self.causal, scale=self.softmax_scale,
        )


def attention(q, k, v, causal=True):
    return scaled_dot_product_attention(q, k, v, is_causal=causal)


__all__ = ["TritonFlashAttention", "attention", "scaled_dot_product_attention", "flash_attn_func", "flash_attn_varlen_func", "FlashAttention"]
