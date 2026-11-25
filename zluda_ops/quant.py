"""Weight-only quantization: int4/int8 groupwise storage with fused dequant.

Storage format (GPTQ/AWQ compatible in spirit):

* symmetric groupwise quantization along K, group size 32/64/128;
* int4: scale = max_abs/7, q = round(w/scale) clamped to [-8, 7], two signed
  nibbles per byte: byte j holds k=2j in the low nibble and k=2j+1 in the high
  nibble (two's complement);
* int8: scale = max_abs/127, one int8 per weight;
* scales are stored as fp16, one per (row, group).

`int4_linear` is a fused Triton GEMV (M=1..16) that dequantizes weights on the
fly, so the fp16 weight matrix is never materialised: the decode path reads
0.5 byte per weight instead of 2. `dequantize_int4/int8` are reference
helpers (used by `dequant_linear`) and are useful for correctness checks;
`linear_quantized` is not memory-efficient by itself.
"""

import torch
import triton
import triton.language as tl


# Tuned on gfx1010 (N=K=4096, GROUP=64): (BK, SPLIT_K, ACC_MODE, num_warps).
# ACC_MODE: 0 = fp32 products+acc, 1 = fp16 products + fp32 acc (fastest),
# 2 = fully fp16 accumulation (same speed, worse accuracy -> not used).
# SPLIT_K=1 wins: the atomic-add reduction costs more than the extra parallelism.
CONFIGS = {
    1: (4096, 1, 1, 4),
    2: (1024, 1, 1, 4),
    4: (1024, 1, 1, 4),
    8: (1024, 1, 1, 4),
    16: (512, 1, 1, 4),
}


@triton.jit
def _gemv_int4_kernel(wq_ptr, sc_ptr, x_ptr, b_ptr, y_ptr, N, K,
                      M: tl.constexpr, GROUP: tl.constexpr, BK: tl.constexpr,
                      SPLIT_K: tl.constexpr, ACC_MODE: tl.constexpr,
                      HAS_BIAS: tl.constexpr):
    row = tl.program_id(0)
    pid_k = tl.program_id(1)
    ncols = tl.arange(0, M)
    acc = tl.zeros((M,), tl.float32)
    acc16 = tl.zeros((M,), tl.float16)
    ngroups = K // GROUP
    ngrp_chunk: tl.constexpr = BK // GROUP
    k_per = K // SPLIT_K
    for k0 in range(pid_k * k_per, (pid_k + 1) * k_per, BK):
        offs_b = k0 // 2 + tl.arange(0, BK // 2)
        packed = tl.load(wq_ptr + row * (K // 2) + offs_b).to(tl.int32)
        lo = ((packed & 0xF) ^ 8) - 8
        hi = (((packed >> 4) & 0xF) ^ 8) - 8
        # One scale load per group, broadcast to bytes instead of a per-byte
        # gather (32x fewer loads for GROUP=64).
        sc_g = tl.load(sc_ptr + row * ngroups + (k0 // GROUP + tl.arange(0, ngrp_chunk)))
        sc_vec = tl.reshape(
            tl.broadcast_to(sc_g[:, None], (ngrp_chunk, GROUP // 2)),
            (BK // 2,),
        )
        xc = tl.load(x_ptr + ncols[:, None] * K + (k0 + tl.arange(0, BK))[None, :])
        xr = tl.reshape(xc, (M, BK // 2, 2))
        xlo, xhi = tl.split(xr)
        if ACC_MODE == 2:
            # Fully fp16: products and the per-chunk sum stay in fp16, only the
            # per-chunk partial is promoted to fp32 across iterations.
            sf = sc_vec.to(tl.float16)
            wlo = lo.to(tl.float16) * sf
            whi = hi.to(tl.float16) * sf
            prod16 = wlo[None, :] * xlo + whi[None, :] * xhi
            acc += tl.sum(prod16, axis=1).to(tl.float32)
        elif ACC_MODE == 1:
            # fp16 products, fp32 accumulation (like the fp16 HPA GEMM path).
            sf = sc_vec.to(tl.float16)
            wlo = lo.to(tl.float16) * sf
            whi = hi.to(tl.float16) * sf
            prod = (wlo[None, :] * xlo + whi[None, :] * xhi).to(tl.float32)
            acc += tl.sum(prod, axis=1)
        else:
            # fp32 products and accumulation (reference mode).
            scf = sc_vec[None, :]
            wlo = lo.to(tl.float32) * scf
            whi = hi.to(tl.float32) * scf
            prod = wlo * xlo.to(tl.float32) + whi * xhi.to(tl.float32)
            acc += tl.sum(prod, axis=1)
    if HAS_BIAS and pid_k == 0:
        acc += tl.load(b_ptr + row).to(tl.float32)
    if SPLIT_K == 1:
        tl.store(y_ptr + ncols * N + row, acc.to(y_ptr.dtype.element_ty))
    else:
        tl.atomic_add(y_ptr + ncols * N + row, acc)


def quantize_int4(weight, group_size=64):
    """Symmetric int4 groupwise quantization. Returns (packed uint8, scales fp16)."""
    w = weight.detach().float()
    n, k = w.shape
    if k % group_size or k % 2 or w.shape != weight.shape:
        raise ValueError("K must be divisible by group_size and 2")
    g = w.reshape(n, k // group_size, group_size)
    max_abs = g.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    scale = (max_abs / 7.0).to(torch.float16)
    q = torch.round(g / scale.float()).clamp(-8, 7).to(torch.int16).reshape(n, k)
    low = (q[:, 0::2] & 0xF).to(torch.uint8)
    high = ((q[:, 1::2] & 0xF) << 4).to(torch.uint8)
    packed = (low | high).contiguous()
    return packed, scale.reshape(n, k // group_size).contiguous()


def dequantize_int4(packed, scales, shape, group_size=64):
    n, k = shape
    b = packed.to(torch.int16)
    q = torch.empty((n, k), dtype=torch.int16, device=packed.device)
    q[:, 0::2] = b & 0xF
    q[:, 1::2] = (b >> 4) & 0xF
    q = ((q ^ 8) - 8).to(torch.float32)
    w = q.reshape(n, k // group_size, group_size) * scales.float().unsqueeze(-1)
    return w.reshape(n, k).to(torch.float16)


def quantize_int8(weight, group_size=128):
    """Symmetric int8 groupwise quantization. Returns (int8 weights, fp16 scales)."""
    w = weight.detach().float()
    n, k = w.shape
    if k % group_size:
        raise ValueError("K must be divisible by group_size")
    g = w.reshape(n, k // group_size, group_size)
    max_abs = g.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    scale = (max_abs / 127.0).to(torch.float16)
    q = torch.round(g / scale.float()).clamp(-127, 127).to(torch.int8).reshape(n, k)
    return q.contiguous(), scale.reshape(n, k // group_size).contiguous()


def dequantize_int8(packed, scales, shape, group_size=128):
    n, k = shape
    w = packed.to(torch.float32).reshape(n, k // group_size, group_size)
    w = w * scales.float().unsqueeze(-1)
    return w.reshape(n, k).to(torch.float16)


def _supported(x, packed, scales, bias, group_size):
    if not x.is_cuda or x.dim() != 2:
        return False
    if x.dtype != torch.float16 or not x.is_contiguous():
        return False
    m = x.shape[0]
    if m not in CONFIGS:
        return False
    n, k = packed.shape[0], packed.shape[1] * 2
    if x.shape[1] != k or k % group_size or k % 2:
        return False
    if packed.dtype != torch.uint8 or not packed.is_contiguous():
        return False
    if packed.shape != (n, k // 2):
        return False
    if scales.dtype != torch.float16 or not scales.is_contiguous():
        return False
    if scales.shape != (n, k // group_size):
        return False
    if bias is not None and (bias.dtype != torch.float16 or not bias.is_contiguous()
                             or bias.shape != (n,)):
        return False
    return True


def _int4_linear_config(x, packed, scales, bias, group_size, bk, splits, acc_mode, warps):
    n = packed.shape[0]
    k = packed.shape[1] * 2
    m = x.shape[0]
    if splits == 1:
        y = torch.empty(m, n, device=x.device, dtype=x.dtype)
        y_arg = y
    else:
        y_arg = torch.zeros(m, n, device=x.device, dtype=torch.float32)
    _gemv_int4_kernel[(n, splits)](
        packed, scales, x, bias if bias is not None else y_arg, y_arg, n, k,
        M=m, GROUP=group_size, BK=bk, SPLIT_K=splits, ACC_MODE=acc_mode,
        HAS_BIAS=bias is not None, num_warps=warps, num_stages=1,
    )
    return y_arg if splits == 1 else y_arg.to(x.dtype)


def int4_linear(x, packed, scales, bias=None, group_size=64):
    """Fused dequant+GEMV for M=1..16. Returns fp16 (M, N) or None if unsupported."""
    if not _supported(x, packed, scales, bias, group_size):
        return None
    m = x.shape[0]
    bk, splits, acc_mode, warps = CONFIGS[m]
    k = packed.shape[1] * 2
    # BK must be a power of two that divides K; shrink it for unusual K.
    while bk > 128 and k % bk:
        bk //= 2
    if k % bk:
        return None
    return _int4_linear_config(x, packed, scales, bias, group_size, bk, splits,
                               acc_mode, warps)


def linspace_reference(x, packed, scales, bias=None, group_size=64):
    """Correctness reference: dequantize to fp16, then plain linear."""
    n, k = packed.shape[0], packed.shape[1] * 2
    w = dequantize_int4(packed, scales, (n, k), group_size)
    return torch.nn.functional.linear(x, w, bias)


__all__ = [
    "quantize_int4", "dequantize_int4", "quantize_int8", "dequantize_int8",
    "int4_linear", "linspace_reference", "CONFIGS",
]
