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
    bk, warps = CONFIGS[m]
    y = torch.empty(m, n, device=x.device, dtype=x.dtype)
    _gemv_kernel[(n,)](w, x, bias if bias is not None else y, y, n, k, M=m, BK=bk,
                       HAS_BIAS=bias is not None, PROD_F16=not accurate,
                       num_warps=warps, num_stages=1)
    return y
