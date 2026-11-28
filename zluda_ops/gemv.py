import torch
import triton
import triton.language as tl

CONFIGS = {
    1: (4096, 4),
    2: (4096, 4),
    4: (2048, 8),
    8: (4096, 8),
    16: (4096, 4),
    32: (4096, 4),
}

# Shape-aware (BK, num_warps) overrides tuned on gfx1010 for M=1 decode
# (2026-11-25 sweep): qkv/o 896x896 (512,2) 2.07x rocBLAS, gate/up 4864x896
# (1024,2) 2.01x, down 896x4864 (512,4) 5.29x, lm_head 151936x896 (1024,2)
# 12.2x. Falls back to CONFIGS[m] for other shapes.
TUNED_M1 = {
    (896, 896): (512, 2),
    (4864, 896): (1024, 2),
    (896, 4864): (512, 4),
    (151936, 896): (1024, 2),
}


def _config(m, n, k):
    if m == 1:
        cfg = TUNED_M1.get((n, k))
        if cfg is not None:
            return cfg
        try:
            from .tuned_gemv_ext import TUNED_M1_EXT

            cfg = TUNED_M1_EXT.get((n, k))
        except Exception:  # noqa: BLE001 - generated table is optional
            cfg = None
        if cfg is not None:
            return cfg
    return CONFIGS[m]


@triton.jit
def _gemv_kernel(w_ptr, x_ptr, b_ptr, y_ptr, N, K, M: tl.constexpr, BK: tl.constexpr,
                 HAS_BIAS: tl.constexpr, PROD_F16: tl.constexpr):
    row = tl.program_id(0)
    mcols = tl.arange(0, M)
    acc = tl.zeros((M,), tl.float32)
    for k0 in range(0, K, BK):
        offs = k0 + tl.arange(0, BK)
        kmask = offs < K
        w = tl.load(w_ptr + row * K + offs, mask=kmask, other=0.0)
        x = tl.load(x_ptr + mcols[:, None] * K + offs[None, :], mask=kmask[None, :], other=0.0)
        if PROD_F16:
            acc += tl.sum(w[None, :] * x, axis=1).to(tl.float32)
        else:
            acc += tl.sum(w[None, :].to(tl.float32) * x.to(tl.float32), axis=1)
    if HAS_BIAS:
        acc += tl.load(b_ptr + row).to(tl.float32)
    tl.store(y_ptr + mcols * N + row, acc.to(y_ptr.dtype.element_ty))


def _supported(x, w, bias):
    if not x.is_cuda or x.dim() != 2 or w.dim() != 2:
        return False
    if x.dtype not in (torch.float16, torch.float32) or w.dtype != x.dtype:
        return False
    if not x.is_contiguous() or not w.is_contiguous():
        return False
    if x.shape[0] not in CONFIGS or x.shape[1] != w.shape[1]:
        return False
    if bias is not None and (bias.dtype != x.dtype or not bias.is_contiguous()):
        return False
    return True


def linear(x, w, bias=None, accurate=True):
    if not _supported(x, w, bias):
        return None
    n, k = w.shape
    m = x.shape[0]
    bk, warps = _config(m, n, k)
    y = torch.empty(m, n, device=x.device, dtype=x.dtype)
    _gemv_kernel[(n,)](w, x, bias if bias is not None else y, y, n, k, M=m, BK=bk,
                       HAS_BIAS=bias is not None, PROD_F16=not accurate,
                       num_warps=warps, num_stages=1)
    return y
