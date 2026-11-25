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


# BK = number of weights processed per loop iteration (must divide K).
# M>=4 uses a smaller BK to keep the (M, BK/2) fp32 temporaries in registers.
CONFIGS = {
    1: (2048, 4),
    2: (1024, 4),
    4: (512, 4),
    8: (512, 4),
    16: (512, 8),
}


@triton.jit
def _gemv_int4_kernel(wq_ptr, sc_ptr, x_ptr, b_ptr, y_ptr, N, K,
                      M: tl.constexpr, GROUP: tl.constexpr, BK: tl.constexpr,
                      HAS_BIAS: tl.constexpr):
    row = tl.program_id(0)
    ncols = tl.arange(0, M)
    acc = tl.zeros((M,), tl.float32)
    ngroups = K // GROUP
    for k0 in range(0, K, BK):
        offs_b = k0 // 2 + tl.arange(0, BK // 2)
        packed = tl.load(wq_ptr + row * (K // 2) + offs_b).to(tl.int32)
        lo = (packed & 0xF) ^ 8
        hi = ((packed >> 4) & 0xF) ^ 8
        lo = lo - 8
        hi = hi - 8
        # Both nibbles of a byte belong to the same group for even GROUP.
        g = (2 * offs_b) // GROUP
        scale = tl.load(sc_ptr + row * ngroups + g).to(tl.float32)
        wlo = lo.to(tl.float32) * scale
        whi = hi.to(tl.float32) * scale
        # Load the x chunk contiguously and split adjacent fp16 pairs instead of
        # two strided loads: fewer memory transactions, no stride-2 access.
        xc = tl.load(x_ptr + ncols[:, None] * K + (k0 + tl.arange(0, BK))[None, :])
        xr = tl.reshape(xc, (M, BK // 2, 2))
        xlo, xhi = tl.split(xr)
        acc += tl.sum(wlo[None, :] * xlo.to(tl.float32)
                      + whi[None, :] * xhi.to(tl.float32), axis=1)
    if HAS_BIAS:
        acc += tl.load(b_ptr + row).to(tl.float32)
    tl.store(y_ptr + ncols * N + row, acc.to(y_ptr.dtype.element_ty))


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


def int4_linear(x, packed, scales, bias=None, group_size=64):
    """Fused dequant+GEMV for M=1..16. Returns fp16 (M, N) or None if unsupported."""
    if not _supported(x, packed, scales, bias, group_size):
        return None
    m = x.shape[0]
    k = packed.shape[1] * 2
    n = packed.shape[0]
    bk, warps = CONFIGS[m]
    if k % bk:
        bk = k
        warps = 4
    y = torch.empty(m, n, device=x.device, dtype=x.dtype)
    _gemv_int4_kernel[(n,)](
        packed, scales, x, bias if bias is not None else y, y, n, k,
        M=m, GROUP=group_size, BK=bk, HAS_BIAS=bias is not None,
        num_warps=warps, num_stages=1,
    )
    return y


def linspace_reference(x, packed, scales, bias=None, group_size=64):
    """Correctness reference: dequantize to fp16, then plain linear."""
    n, k = packed.shape[0], packed.shape[1] * 2
    w = dequantize_int4(packed, scales, (n, k), group_size)
    return torch.nn.functional.linear(x, w, bias)


__all__ = [
    "quantize_int4", "dequantize_int4", "quantize_int8", "dequantize_int8",
    "int4_linear", "linspace_reference", "CONFIGS",
]
