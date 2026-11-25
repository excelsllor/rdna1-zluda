"""8-bit AdamW optimizer (bitsandbytes-style) implemented with Triton.

bitsandbytes' prebuilt CUDA kernels cannot run through ZLUDA on gfx1010
(`named symbol not found` in its nvcc-compiled ops), so this module provides an
8-bit optimizer with the same idea: the first and second moment estimates are
stored as int8 per block with an fp32 absmax scale per block, while parameters
and gradients stay fp32. That cuts optimizer-state memory from 8 bytes to
~2.03 bytes per parameter (int8 + int8 + 2 fp32 scales per block of 256).

Usage:
    opt = AdamW8bit(model.parameters(), lr=1e-4, weight_decay=0.01)
    loss.backward()
    opt.step()          # torch-style
    opt.zero_grad()

Supported: fp32 CUDA (ZLUDA) parameters. The bit-exactness of the fp32 AdamW
trajectory is not preserved (states are quantized), as in bitsandbytes.
"""

import torch
import triton
import triton.language as tl

BLOCK = 256


@triton.jit
def _adamw8bit_kernel(P, G, MQ, VQ, MABS, VABS, N, lr, b1, b2, bc1, bc2, eps,
                      WD, BLOCK: tl.constexpr, HAS_WD: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    p = tl.load(P + offs, mask=mask, other=0.0).to(tl.float32)
    g = tl.load(G + offs, mask=mask, other=0.0).to(tl.float32)
    mq = tl.load(MQ + offs, mask=mask, other=0).to(tl.int32)
    vq = tl.load(VQ + offs, mask=mask, other=0).to(tl.int32)
    mabs = tl.load(MABS + pid)
    vabs = tl.load(VABS + pid)
    # dequantize states to fp32
    m = mq.to(tl.float32) * (mabs / 127.0)
    v = vq.to(tl.float32) * (vabs / 127.0)
    # AdamW update in fp32
    m = m * b1 + g * (1.0 - b1)
    v = v * b2 + g * g * (1.0 - b2)
    mhat = m / bc1
    vhat = v / bc2
    if HAS_WD:
        p = p - lr * (mhat / (tl.sqrt(vhat) + eps) + WD * p)
    else:
        p = p - lr * (mhat / (tl.sqrt(vhat) + eps))
    # requantize states to int8 with a per-block absmax
    m_amax = tl.max(tl.abs(m))
    v_amax = tl.max(tl.abs(v))
    m_amax = tl.maximum(m_amax, 1e-12)
    v_amax = tl.maximum(v_amax, 1e-12)
    mq_new = tl.maximum(tl.minimum(tl.round(m / m_amax * 127.0), 127.0), -127.0)
    vq_new = tl.maximum(tl.minimum(tl.round(v / v_amax * 127.0), 127.0), -127.0)
    tl.store(P + offs, p.to(P.dtype.element_ty), mask=mask)
    tl.store(MQ + offs, mq_new.to(tl.int8), mask=mask)
    tl.store(VQ + offs, vq_new.to(tl.int8), mask=mask)
    tl.store(MABS + pid, m_amax)
    tl.store(VABS + pid, v_amax)


class AdamW8bit:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.01):
        self.params = [p for p in params if p.requires_grad]
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.step_num = 0
        self.state = {}
        self._total_elems = 0
        for p in self.params:
            data = p.data if p.data.is_cuda else p
            flat = data.reshape(-1)
            if flat.dtype != torch.float32 or not flat.is_cuda or not flat.is_contiguous():
                raise NotImplementedError("AdamW8bit supports contiguous fp32 CUDA parameters")
            n = flat.numel()
            nblocks = (n + BLOCK - 1) // BLOCK
            self.state[id(p)] = {
                "mq": torch.zeros(n, device=flat.device, dtype=torch.int8),
                "vq": torch.zeros(n, device=flat.device, dtype=torch.int8),
                "mabs": torch.zeros(nblocks, device=flat.device, dtype=torch.float32),
                "vabs": torch.zeros(nblocks, device=flat.device, dtype=torch.float32),
            }
            self._total_elems += n

    @torch.no_grad()
    def step(self, closure=None):
        self.step_num += 1
        bc1 = 1.0 - self.beta1 ** self.step_num
        bc2 = 1.0 - self.beta2 ** self.step_num
        for p in self.params:
            if p.grad is None:
                continue
            g = p.grad
            if g.dtype != torch.float32:
                g = g.float()
            if not g.is_contiguous():
                g = g.contiguous()
            flat_p = p.data.reshape(-1)
            flat_g = g.reshape(-1)
            st = self.state[id(p)]
            n = flat_p.numel()
            _adamw8bit_kernel[(triton.cdiv(n, BLOCK),)](
                flat_p, flat_g, st["mq"], st["vq"], st["mabs"], st["vabs"], n,
                float(self.lr), float(self.beta1), float(self.beta2),
                float(bc1), float(bc2), float(self.eps), float(self.weight_decay),
                BLOCK=BLOCK, HAS_WD=self.weight_decay > 0.0, num_warps=4,
            )

    def zero_grad(self, set_to_none=True):
        for p in self.params:
            if p.grad is not None:
                if set_to_none:
                    p.grad = None
                else:
                    p.grad.zero_()

    def state_memory_bytes(self):
        return sum(st["mq"].numel() + st["vq"].numel()
                   + st["mabs"].numel() * 4 + st["vabs"].numel() * 4
                   for st in self.state.values())

    def fp32_state_memory_bytes(self):
        return self._total_elems * 8


__all__ = ["AdamW8bit"]
