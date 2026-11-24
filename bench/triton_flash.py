import math
import time

import torch
import triton
import triton.language as tl


@triton.jit
def attn_fwd(
    Q, K, V, Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    Z, H, N_CTX,
    scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    qvk_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh

    Q_block_ptr = tl.make_block_ptr(
        base=Q + qvk_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + off_z.to(tl.int64) * stride_kz + off_h.to(tl.int64) * stride_kh,
        shape=(HEAD_DIM, N_CTX),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_N),
        order=(0, 1),
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + off_z.to(tl.int64) * stride_vz + off_h.to(tl.int64) * stride_vh,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_vn, stride_vk),
        offsets=(0, 0),
        block_shape=(BLOCK_N, HEAD_DIM),
        order=(1, 0),
    )
    O_block_ptr = tl.make_block_ptr(
        base=Out + qvk_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_om, stride_ok),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )

    q = tl.load(Q_block_ptr)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    if CAUSAL:
        hi = tl.cdiv((start_m + 1) * BLOCK_M, BLOCK_N)
    else:
        hi = tl.cdiv(N_CTX, BLOCK_N)

    for start_n in range(0, hi):
        k = tl.load(K_block_ptr)
        qk = tl.dot(q, k)
        if CAUSAL:
            qk = tl.where(
                start_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
                >= start_n * BLOCK_N + tl.arange(0, BLOCK_N)[None, :],
                qk,
                float("-inf"),
            )
        m_new = tl.maximum(m_i, tl.max(qk, 1) * scale)
        qk = qk * scale - m_new[:, None]
        p = tl.math.exp2(qk * 1.44269504)
        l_i = l_i * tl.math.exp2((m_i - m_new) * 1.44269504) + tl.sum(p, 1)
        v = tl.load(V_block_ptr)
        acc = acc * tl.math.exp2((m_i - m_new) * 1.44269504)[:, None]
        acc += tl.dot(p.to(tl.float16), v)
        m_i = m_new
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))

    acc = acc / l_i[:, None]
    tl.store(O_block_ptr, acc.to(tl.float16))


def run(q, k, v, causal):
    B, H, S, D = q.shape
    o = torch.empty_like(q)
    grid = (triton.cdiv(S, 64), B * H)
    attn_fwd[grid](
        q, k, v, o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        B, H, S,
        scale=1.0 / math.sqrt(D),
        BLOCK_M=64, BLOCK_N=64, HEAD_DIM=D,
        CAUSAL=causal,
        num_warps=4, num_stages=1,
    )
    return o


def bench(fn, warmup=3, iters=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000.0


torch.manual_seed(0)
for causal in (False, True):
    for s in (512, 1024, 2048):
        q = torch.randn(1, 8, s, 64, device="cuda", dtype=torch.float16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        try:
            o = run(q, k, v, causal)
            torch.cuda.synchronize()
            from torch.nn.attention import SDPBackend, sdpa_kernel
            with sdpa_kernel(SDPBackend.MATH):
                ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=causal)
            err = (o.float() - ref.float()).abs().max().item()
            ms = bench(lambda: run(q, k, v, causal))
            flops = 4 * s * s * 64 * 8 * (0.5 if causal else 1.0)
            tflops = flops / (ms / 1000) / 1e12
            print(f"triton flash causal={causal} s={s}: {ms:8.2f} ms  {tflops:6.2f} TFLOPS  max err {err:.4f}")
        except Exception as e:
            print(f"triton flash causal={causal} s={s}: FAIL {type(e).__name__}: {str(e)[:150]}")
