"""Fused Triton kernels for common transformer operations on gfx1010.

Implemented (fp16 tensors, contiguous last dimension):

* `rms_norm(x, weight, eps)` with forward+backward (row-wise, one pass);
* `rope_apply(x, cos, sin)` - LLaMA-style rotate-half RoPE with backward;
* `swiglu(gate, up)` - fused SiLU(gate)*up with backward;
* `softmax(x)` - row softmax forward;
* `chunked_cross_entropy(logits, targets, chunk)` - memory-saving CE that never
  materialises the full (N, V) log-softmax.

Each op is written for the shapes that TorchInductor does not fuse well on
RDNA1 (small hidden sizes, memory-bound pointwise chains). Verify with
`bench/fused.py`; numbers are in RESULTS_RDNA1.md.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _rms_norm_fwd(X, W, Y, RSTD, N: tl.constexpr, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + row * N + offs, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(W + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(Y + row * N + offs, (x * rstd * w).to(Y.dtype.element_ty), mask=mask)
    tl.store(RSTD + row, rstd)


@triton.jit
def _rms_norm_bwd(DY, X, W, RSTD, DX, DW, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + row * N + offs, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(DY + row * N + offs, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W + offs, mask=mask, other=0.0).to(tl.float32)
    rstd = tl.load(RSTD + row).to(tl.float32)
    wdyx = tl.sum(dy * w * x, axis=0)
    dx = rstd * dy * w - (rstd * rstd * rstd / N) * x * wdyx
    tl.store(DX + row * N + offs, dx.to(DX.dtype.element_ty), mask=mask)
    tl.atomic_add(DW + offs, dy * x * rstd, mask=mask)


class _RMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps):
        rows, n = x.reshape(-1, x.shape[-1]).shape
        block = triton.next_power_of_2(n)
        y = torch.empty_like(x)
        rstd = torch.empty(rows, device=x.device, dtype=torch.float32)
        _rms_norm_fwd[(rows,)](x, weight, y, rstd, n, float(eps), BLOCK=block,
                               num_warps=4, num_stages=1)
        ctx.save_for_backward(x, weight, rstd)
        ctx.rows, ctx.n, ctx.block = rows, n, block
        return y

    @staticmethod
    def backward(ctx, dy):
        x, weight, rstd = ctx.saved_tensors
        rows, n, block = ctx.rows, ctx.n, ctx.block
        dx = torch.empty_like(x)
        dw = torch.zeros_like(weight, dtype=torch.float32)
        _rms_norm_bwd[(rows,)](dy, x, weight, rstd, dx, dw, n, BLOCK=block,
                               num_warps=4, num_stages=1)
        return dx, dw.to(weight.dtype), None


def rms_norm(x, weight, eps=1e-6):
    if x.dtype != torch.float16 or not x.is_contiguous() or not x.is_cuda:
        raise NotImplementedError("rms_norm supports contiguous fp16 CUDA tensors")
    return _RMSNorm.apply(x, weight, eps)


@triton.jit
def _rope_fwd(X, COS, SIN, Y, S, D: tl.constexpr, HALF: tl.constexpr):
    row = tl.program_id(0)
    s = row % S
    offs = tl.arange(0, HALF)
    cos = tl.load(COS + s * HALF + offs).to(tl.float32)
    sin = tl.load(SIN + s * HALF + offs).to(tl.float32)
    x1 = tl.load(X + row * D + offs).to(tl.float32)
    x2 = tl.load(X + row * D + HALF + offs).to(tl.float32)
    tl.store(Y + row * D + offs, (x1 * cos - x2 * sin).to(Y.dtype.element_ty))
    tl.store(Y + row * D + HALF + offs, (x2 * cos + x1 * sin).to(Y.dtype.element_ty))


@triton.jit
def _rope_bwd(DY, COS, SIN, DX, S, D: tl.constexpr, HALF: tl.constexpr):
    row = tl.program_id(0)
    s = row % S
    offs = tl.arange(0, HALF)
    cos = tl.load(COS + s * HALF + offs).to(tl.float32)
    sin = tl.load(SIN + s * HALF + offs).to(tl.float32)
    y1 = tl.load(DY + row * D + offs).to(tl.float32)
    y2 = tl.load(DY + row * D + HALF + offs).to(tl.float32)
    tl.store(DX + row * D + offs, (y1 * cos + y2 * sin).to(DX.dtype.element_ty))
    tl.store(DX + row * D + HALF + offs, (-y1 * sin + y2 * cos).to(DX.dtype.element_ty))


class _RoPE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, cos, sin, seq):
        d = x.shape[-1]
        half = d // 2
        y = torch.empty_like(x)
        rows = x.numel() // d
        _rope_fwd[(rows,)](x, cos, sin, y, seq, D=d, HALF=half, num_warps=4)
        ctx.save_for_backward(cos, sin)
        ctx.seq, ctx.d, ctx.half, ctx.rows = seq, d, half, rows
        return y

    @staticmethod
    def backward(ctx, dy):
        cos, sin = ctx.saved_tensors
        dx = torch.empty_like(dy)
        _rope_bwd[(ctx.rows,)](dy, cos, sin, dx, ctx.seq, D=ctx.d, HALF=ctx.half,
                               num_warps=4)
        return dx, None, None, None


def rope_apply(x, cos, sin, seq=None):
    """Rotate-half RoPE for x (B, H, S, D) with cos/sin (S, D/2)."""
    if x.dtype != torch.float16 or not x.is_contiguous() or not x.is_cuda:
        raise NotImplementedError("rope_apply supports contiguous fp16 CUDA tensors")
    seq = x.shape[2] if seq is None else seq
    return _RoPE.apply(x, cos, sin, seq)


@triton.jit
def _swiglu_fwd(G, U, Y, NUM, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < NUM
    g = tl.load(G + offs, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(U + offs, mask=mask, other=0.0).to(tl.float32)
    sig = 1.0 / (1.0 + tl.exp(-g))
    tl.store(Y + offs, (g * sig * u).to(Y.dtype.element_ty), mask=mask)


@triton.jit
def _swiglu_bwd(DY, G, U, DG, DU, NUM, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < NUM
    dy = tl.load(DY + offs, mask=mask, other=0.0).to(tl.float32)
    g = tl.load(G + offs, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(U + offs, mask=mask, other=0.0).to(tl.float32)
    sig = 1.0 / (1.0 + tl.exp(-g))
    silu = g * sig
    dg = dy * u * (sig * (1.0 + g * (1.0 - sig)))
    du = dy * silu
    tl.store(DG + offs, dg.to(DG.dtype.element_ty), mask=mask)
    tl.store(DU + offs, du.to(DU.dtype.element_ty), mask=mask)


class _SwiGLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate, up):
        y = torch.empty_like(gate)
        num = gate.numel()
        block = 1024
        _swiglu_fwd[(triton.cdiv(num, block),)](gate, up, y, num, BLOCK=block,
                                                num_warps=4)
        ctx.save_for_backward(gate, up)
        return y

    @staticmethod
    def backward(ctx, dy):
        gate, up = ctx.saved_tensors
        dg = torch.empty_like(gate)
        du = torch.empty_like(up)
        num = gate.numel()
        block = 1024
        _swiglu_bwd[(triton.cdiv(num, block),)](dy, gate, up, dg, du, num,
                                                BLOCK=block, num_warps=4)
        return dg, du


def swiglu(gate, up):
    if gate.dtype != torch.float16 or not gate.is_cuda or not gate.is_contiguous():
        raise NotImplementedError("swiglu supports contiguous fp16 CUDA tensors")
    return _SwiGLU.apply(gate, up)


@triton.jit
def _softmax_fwd(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + row * N + offs, mask=mask, other=-float("inf")).to(tl.float32)
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    s = tl.sum(e, axis=0)
    tl.store(Y + row * N + offs, (e / s).to(Y.dtype.element_ty), mask=mask)


def softmax(x):
    n = x.shape[-1]
    block = triton.next_power_of_2(n)
    rows = x.numel() // n
    y = torch.empty_like(x)
    _softmax_fwd[(rows,)](x, y, n, BLOCK=block, num_warps=8)
    return y


def chunked_cross_entropy(logits, targets, chunk_size=1024):
    """Mean cross-entropy without materialising the full log-softmax.

    logits: (rows, vocab) any float dtype; targets: (rows,) int64.
    Autograd works through the torch ops used inside; peak memory is
    O(chunk * vocab) instead of O(rows * vocab).
    """
    rows, vocab = logits.shape
    total = torch.zeros((), device=logits.device, dtype=torch.float32)
    for start in range(0, rows, chunk_size):
        end = min(start + chunk_size, rows)
        part = logits[start:end].float()
        tgt = targets[start:end]
        lse = torch.logsumexp(part, dim=-1)
        picked = part.gather(1, tgt[:, None]).squeeze(1)
        total = total + (lse - picked).sum()
    return total / rows


__all__ = ["rms_norm", "rope_apply", "swiglu", "softmax", "chunked_cross_entropy"]
