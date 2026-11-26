"""Fused QKV projection + RoPE + KV-cache append for decode (Phase 2).

Replaces, per transformer layer on the decode path:
  * 3 GEMV launches (q/k/v projections),
  * 2 elementwise RoPE kernels (q and k),
  * 2 KV-cache writes (k/v index_copy),
with a single Triton kernel: one program per head output. For each program the
64 output rows of that head are dot products of x with the corresponding rows of
the concatenated q/k/v weight matrix; q and k heads then get rotate-half RoPE,
and all heads write directly into q_out / the KV cache at `pos`.

Kernel layout assumption: D = 64, weights are the concatenation
    w_qkv = cat([Wq (Hq*64, K), Wk (Hkv*64, K), Wv (Hkv*64, K)], dim=0)
Cache layout: k_cache/v_cache (B, Hkv, N, D) fp16, contiguous; q_out (B, Hq, 1, D),
the kernel writes the single new token at `pos`.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _qkv_rope_append_kernel(X, W, COS, SIN, Q_OUT, K_CACHE, V_CACHE,
                            HQ: tl.constexpr, HKV: tl.constexpr,
                            NCACHE: tl.constexpr, D: tl.constexpr,
                            HALF: tl.constexpr, K: tl.constexpr,
                            BK: tl.constexpr, POS):
    pid = tl.program_id(0)
    if pid < HQ:
        kind = 0
        head = pid
    elif pid < HQ + HKV:
        kind = 1
        head = pid - HQ
    else:
        kind = 2
        head = pid - HQ - HKV
    row0 = pid * D
    offs_h = tl.arange(0, HALF)
    offs = tl.arange(0, BK)
    acc_a = tl.zeros((HALF,), tl.float32)
    acc_b = tl.zeros((HALF,), tl.float32)
    for k0 in range(0, K, BK):
        x = tl.load(X + k0 + offs).to(tl.float32)
        wmask = (k0 + offs)[None, :] < K
        wa = tl.load(W + (row0 + offs_h[:, None]) * K + (k0 + offs)[None, :],
                     mask=wmask, other=0.0).to(tl.float32)
        wb = tl.load(W + (row0 + HALF + offs_h[:, None]) * K + (k0 + offs)[None, :],
                     mask=wmask, other=0.0).to(tl.float32)
        xm = tl.where((k0 + offs) < K, x, 0.0)
        acc_a += tl.sum(wa * xm[None, :], axis=1)
        acc_b += tl.sum(wb * xm[None, :], axis=1)
    if kind == 2:
        dst = head * NCACHE * D + POS * D
        tl.store(V_CACHE + dst + offs_h, acc_a.to(V_CACHE.dtype.element_ty))
        tl.store(V_CACHE + dst + HALF + offs_h, acc_b.to(V_CACHE.dtype.element_ty))
    else:
        cos = tl.load(COS + offs_h).to(tl.float32)
        sin = tl.load(SIN + offs_h).to(tl.float32)
        ro_a = acc_a * cos - acc_b * sin
        ro_b = acc_a * sin + acc_b * cos
        if kind == 0:
            dst = head * D
            tl.store(Q_OUT + dst + offs_h, ro_a.to(Q_OUT.dtype.element_ty))
            tl.store(Q_OUT + dst + HALF + offs_h, ro_b.to(Q_OUT.dtype.element_ty))
        else:
            dst = head * NCACHE * D + POS * D
            tl.store(K_CACHE + dst + offs_h, ro_a.to(K_CACHE.dtype.element_ty))
            tl.store(K_CACHE + dst + HALF + offs_h, ro_b.to(K_CACHE.dtype.element_ty))


def qkv_rope_append(x, w_qkv, cos, sin, k_cache, v_cache, q_out, pos,
                    block_k=512):
    """Fused decode QKV+RoPE+KV-append.

    x: (K,) fp16; w_qkv: ((Hq+2*Hkv)*64, K) fp16; cos/sin: (32,) fp16;
    k_cache/v_cache: (Hkv, N, 64) or (B, Hkv, N, 64) with B==1; q_out:
    (Hq, 1, 64) or (1, Hq, 1, 64). Returns True on success.
    """
    if (not x.is_cuda or x.dtype != torch.float16 or x.dim() != 1
            or w_qkv.dtype != torch.float16):
        return False
    k = x.shape[0]
    if w_qkv.shape[1] != k or w_qkv.shape[0] % 64:
        return False
    total_heads = w_qkv.shape[0] // 64
    ncache = k_cache.shape[-2]
    hkv = k_cache.shape[-3]
    hq = total_heads - 2 * hkv
    if hq <= 0 or q_out.shape[-2] != 1 or q_out.shape[-1] != 64:
        return False
    if cos.numel() != 32 or sin.numel() != 32:
        return False
    x = x.contiguous()
    w_qkv = w_qkv.contiguous()
    q_flat = q_out.reshape(-1)
    c_flat = k_cache.reshape(-1)
    v_flat = v_cache.reshape(-1)
    _qkv_rope_append_kernel[(total_heads,)](
        x, w_qkv, cos.contiguous(), sin.contiguous(), q_flat, c_flat, v_flat,
        HQ=hq, HKV=hkv, NCACHE=ncache, D=64, HALF=32, K=k, BK=block_k,
        POS=int(pos), num_warps=2, num_stages=1,
    )
    return True


__all__ = ["qkv_rope_append"]
