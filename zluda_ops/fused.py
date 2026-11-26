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
from triton.language.extra import libdevice


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




@triton.jit
def _layer_norm_fwd(X, W, B, Y, MEAN, RSTD, N: tl.constexpr, eps, BLOCK: tl.constexpr,
                    HAS_WB: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + row * N + offs, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / N
    xc = x - mean
    var = tl.sum(xc * xc, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    if HAS_WB:
        w = tl.load(W + offs, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(B + offs, mask=mask, other=0.0).to(tl.float32)
    else:
        w = x * 0.0 + 1.0
        b = x * 0.0
    tl.store(Y + row * N + offs, (xc * rstd * w + b).to(Y.dtype.element_ty), mask=mask)
    tl.store(MEAN + row, mean)
    tl.store(RSTD + row, rstd)


@triton.jit
def _layer_norm_bwd(DY, X, W, MEAN, RSTD, DX, DW, DB, N: tl.constexpr,
                    BLOCK: tl.constexpr, HAS_WB: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + row * N + offs, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(DY + row * N + offs, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(MEAN + row)
    rstd = tl.load(RSTD + row)
    xhat = (x - mean) * rstd
    if HAS_WB:
        w = tl.load(W + offs, mask=mask, other=0.0).to(tl.float32)
    else:
        w = tl.zeros((BLOCK,), tl.float32) + 1.0
    dxhat = dy * w
    s1 = tl.sum(dxhat, axis=0)
    s2 = tl.sum(dxhat * xhat, axis=0)
    dx = rstd * (dxhat - s1 / N - xhat * s2 / N)
    tl.store(DX + row * N + offs, dx.to(DX.dtype.element_ty), mask=mask)
    if HAS_WB:
        tl.atomic_add(DW + offs, dy * xhat, mask=mask)
        tl.atomic_add(DB + offs, dy, mask=mask)


class _LayerNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        rows, n = x.reshape(-1, x.shape[-1]).shape
        block = triton.next_power_of_2(n)
        y = torch.empty_like(x)
        mean = torch.empty(rows, device=x.device, dtype=torch.float32)
        rstd = torch.empty(rows, device=x.device, dtype=torch.float32)
        has_wb = weight is not None
        w = weight if has_wb else x
        b = bias if bias is not None else x
        _layer_norm_fwd[(rows,)](x, w, b, y, mean, rstd, n, float(eps),
                                 BLOCK=block, HAS_WB=has_wb, num_warps=4,
                                 num_stages=1)
        ctx.save_for_backward(x, w, mean, rstd)
        ctx.rows, ctx.n, ctx.block, ctx.has_wb = rows, n, block, has_wb
        ctxeps = eps
        return y

    @staticmethod
    def backward(ctx, dy):
        x, w, mean, rstd = ctx.saved_tensors
        dx = torch.empty_like(x)
        dw = torch.zeros_like(w, dtype=torch.float32)
        db = torch.zeros_like(w, dtype=torch.float32)
        _layer_norm_bwd[(ctx.rows,)](dy, x, w, mean, rstd, dx, dw, db, ctx.n,
                                     BLOCK=ctx.block, HAS_WB=ctx.has_wb,
                                     num_warps=4, num_stages=1)
        return dx, dw.to(w.dtype), db.to(w.dtype), None


def layer_norm(x, weight=None, bias=None, eps=1e-5):
    if x.dtype != torch.float16 or not x.is_contiguous() or not x.is_cuda:
        raise NotImplementedError("layer_norm supports contiguous fp16 CUDA tensors")
    return _LayerNorm.apply(x, weight, bias, eps)


@triton.jit
def _geglu_fwd(G, U, Y, NUM, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < NUM
    g = tl.load(G + offs, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(U + offs, mask=mask, other=0.0).to(tl.float32)
    gelu = 0.5 * g * (1.0 + libdevice.erf(g * 0.7071067811865476))
    tl.store(Y + offs, (gelu * u).to(Y.dtype.element_ty), mask=mask)


@triton.jit
def _geglu_bwd(DY, G, U, DG, DU, NUM, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < NUM
    dy = tl.load(DY + offs, mask=mask, other=0.0).to(tl.float32)
    g = tl.load(G + offs, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(U + offs, mask=mask, other=0.0).to(tl.float32)
    cdf = 0.5 * (1.0 + libdevice.erf(g * 0.7071067811865476))
    pdf = tl.exp(-0.5 * g * g) * 0.3989422804014327
    dgelu = cdf + g * pdf
    gelu = g * cdf
    tl.store(DG + offs, (dy * u * dgelu).to(DG.dtype.element_ty), mask=mask)
    tl.store(DU + offs, (dy * gelu).to(DU.dtype.element_ty), mask=mask)


class _GeGLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate, up):
        y = torch.empty_like(gate)
        num = gate.numel()
        block = 1024
        _geglu_fwd[(triton.cdiv(num, block),)](gate, up, y, num, BLOCK=block,
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
        _geglu_bwd[(triton.cdiv(num, block),)](dy, gate, up, dg, du, num,
                                               BLOCK=block, num_warps=4)
        return dg, du


def geglu(gate, up):
    if gate.dtype != torch.float16 or not gate.is_cuda or not gate.is_contiguous():
        raise NotImplementedError("geglu supports contiguous fp16 CUDA tensors")
    return _GeGLU.apply(gate, up)


@triton.jit
def _adamw_kernel(P, G, M, V, NUM, lr, b1, b2, bc1, bc2, eps, wd,
                  BLOCK: tl.constexpr, HAS_WD: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < NUM
    p = tl.load(P + offs, mask=mask, other=0.0).to(tl.float32)
    g = tl.load(G + offs, mask=mask, other=0.0).to(tl.float32)
    m = tl.load(M + offs, mask=mask, other=0.0).to(tl.float32)
    v = tl.load(V + offs, mask=mask, other=0.0).to(tl.float32)
    m = m * b1 + g * (1.0 - b1)
    v = v * b2 + g * g * (1.0 - b2)
    mhat = m / bc1
    vhat = v / bc2
    if HAS_WD:
        p = p - lr * (mhat / (tl.sqrt(vhat) + eps) + wd * p)
    else:
        p = p - lr * (mhat / (tl.sqrt(vhat) + eps))
    tl.store(P + offs, p.to(P.dtype.element_ty), mask=mask)
    tl.store(M + offs, m.to(M.dtype.element_ty), mask=mask)
    tl.store(V + offs, v.to(V.dtype.element_ty), mask=mask)


def adamw_step_(param, grad, exp_avg, exp_avg_sq, lr=1e-3, beta1=0.9,
                beta2=0.999, eps=1e-8, weight_decay=0.01, step=1):
    """In-place AdamW update for fp32 tensors (matches torch.optim.AdamW)."""
    if param.dtype != torch.float32 or not param.is_cuda:
        raise NotImplementedError("adamw_step_ supports fp32 CUDA tensors")
    if not (param.is_contiguous() and grad.is_contiguous()
            and exp_avg.is_contiguous() and exp_avg_sq.is_contiguous()):
        raise NotImplementedError("adamw_step_ needs contiguous tensors")
    num = param.numel()
    block = 1024
    bc1 = 1.0 - beta1 ** step
    bc2 = 1.0 - beta2 ** step
    _adamw_kernel[(triton.cdiv(num, block),)](
        param, grad, exp_avg, exp_avg_sq, num, lr, float(beta1), float(beta2), bc1, bc2, eps,
        float(weight_decay), BLOCK=block, HAS_WD=weight_decay > 0.0,
        num_warps=4,
    )
    return param

__all__ = ["add_rms_norm", "rms_norm", "rope_apply", "swiglu", "swiglu_gemv", "softmax", "chunked_cross_entropy", "layer_norm", "geglu", "adamw_step_"]

# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Fused gate/up + SwiGLU GEMV (decode): y[n] = silu(x.Wg[n]+bg) * (x.Wu[n]+bu)
# ---------------------------------------------------------------------------
@triton.jit
def _swiglu_gemv_kernel(WG, WU, X, Y, BG, BU, N, K,
                        BLOCK_N: tl.constexpr, BK: tl.constexpr,
                        HAS_BIAS: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    rmask = rows < N
    ag = tl.zeros((BLOCK_N,), tl.float32)
    au = tl.zeros((BLOCK_N,), tl.float32)
    for k0 in range(0, K, BK):
        offs = k0 + tl.arange(0, BK)
        kmask = offs < K
        x = tl.load(X + offs, mask=kmask, other=0.0).to(tl.float32)
        wmask = rmask[:, None] & kmask[None, :]
        wg = tl.load(WG + rows[:, None] * K + offs[None, :],
                     mask=wmask, other=0.0).to(tl.float32)
        wu = tl.load(WU + rows[:, None] * K + offs[None, :],
                     mask=wmask, other=0.0).to(tl.float32)
        ag += tl.sum(wg * x[None, :], axis=1)
        au += tl.sum(wu * x[None, :], axis=1)
    if HAS_BIAS:
        ag += tl.load(BG + rows, mask=rmask, other=0.0).to(tl.float32)
        au += tl.load(BU + rows, mask=rmask, other=0.0).to(tl.float32)
    tl.store(Y + rows, ((ag * tl.sigmoid(ag)) * au).to(Y.dtype.element_ty),
             mask=rmask)


def swiglu_gemv(x, w_gate, w_up, bias_gate=None, bias_up=None,
                block_n=64, block_k=256):
    """Fused silu(x@Wg + bg) * (x@Wu + bu) for M=1 decode. Returns (N,) fp16."""
    if (not x.is_cuda or x.dtype != torch.float16 or x.dim() != 2
            or x.shape[0] != 1 or not x.is_contiguous()):
        return None
    n, k = w_gate.shape
    if w_up.shape != (n, k) or not w_gate.is_contiguous() or not w_up.is_contiguous():
        return None
    had_bias = bias_gate is not None and bias_up is not None
    if had_bias and (bias_gate.shape != (n,) or bias_up.shape != (n,)
                     or bias_gate.dtype != torch.float16
                     or bias_up.dtype != torch.float16):
        return None
    y = torch.empty(n, device=x.device, dtype=torch.float16)
    grid = (triton.cdiv(n, block_n),)
    _swiglu_gemv_kernel[grid](
        w_gate, w_up, x, y,
        bias_gate if had_bias else y, bias_up if had_bias else y,
        n, k, BLOCK_N=block_n, BK=block_k, HAS_BIAS=had_bias,
        num_warps=4, num_stages=1,
    )
    return y

# ---------------------------------------------------------------------------
# Fused residual add + RMSNorm: y = rms_norm(x + delta) * weight, and the
# pre-norm residual (x + delta) is returned too because the next layer needs it.
# Replaces two bandwidth-bound kernels (add, rms_norm) per transformer layer.
# ---------------------------------------------------------------------------
@triton.jit
def _add_rms_norm_kernel(X, D, W, Y, R, N, BLOCK: tl.constexpr, eps):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + row * N + offs, mask=mask, other=0.0).to(tl.float32)
    d = tl.load(D + row * N + offs, mask=mask, other=0.0).to(tl.float32)
    res = x + d
    ms = tl.sum(tl.where(mask, res * res, 0.0)) / N
    rstd = 1.0 / tl.sqrt(ms + eps)
    w = tl.load(W + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(Y + row * N + offs, (res * rstd * w).to(Y.dtype.element_ty), mask=mask)
    tl.store(R + row * N + offs, res.to(R.dtype.element_ty), mask=mask)


def add_rms_norm(x, delta, weight, eps=1e-6):
    """Fused residual add + RMSNorm. Returns (normed, residual) with x's dtype."""
    n = x.shape[-1]
    if (not x.is_cuda or x.dtype != torch.float16 or x.shape != delta.shape
            or weight.shape != (n,) or weight.dtype != torch.float16):
        return None
    if n > 8192:
        return None
    xf = x.reshape(-1, n)
    df = delta.reshape(-1, n)
    rows = xf.shape[0]
    out = torch.empty_like(xf)
    res = torch.empty_like(xf)
    block = max(128, triton.next_power_of_2(n))
    _add_rms_norm_kernel[(rows,)](xf, df, weight, out, res, n, BLOCK=block,
                                  eps=eps, num_warps=4, num_stages=1)
    return out.reshape(x.shape), res.reshape(x.shape)
