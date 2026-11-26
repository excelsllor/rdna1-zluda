"""bitsandbytes.functional shim: blockwise int8 helpers built on torch."""

import torch

__all__ = [
    "quantize_blockwise", "dequantize_blockwise", "int8_vectorwise_quant",
    "int8_vectorwise_dequant", "int8_linear_matmul", "int8_mm_dequant",
]


def quantize_blockwise(A, code=None, blocksize=4096, absmax=None, out=None):
    """Blockwise absolute-max int8 quantization (simplified bnb semantics)."""
    flat = A.reshape(-1).float()
    n = flat.numel()
    nb = (n + blocksize - 1) // blocksize
    padded = torch.zeros(nb * blocksize, device=A.device, dtype=torch.float32)
    padded[:n] = flat
    blocks = padded.reshape(nb, blocksize)
    amax = blocks.abs().amax(dim=1).clamp(min=1e-12)
    q = torch.round(blocks / amax[:, None] * 127.0).clamp(-127, 127).to(torch.int8)
    return q.reshape(-1)[:n].contiguous(), amax


def dequantize_blockwise(A, absmax, blocksize=4096, out=None):
    n = A.numel()
    nb = absmax.numel()
    padded = torch.zeros(nb * blocksize, device=A.device, dtype=torch.float32)
    padded[:n] = A.float()
    blocks = padded.reshape(nb, blocksize) * absmax[:, None]
    return blocks.reshape(-1)[:n].to(torch.float32)


def int8_vectorwise_quant(A, threshold=0.0):
    amax = A.abs().amax().clamp(min=1e-12)
    q = torch.round(A.float() / amax * 127.0).clamp(-127, 127).to(torch.int8)
    return q, amax


def int8_vectorwise_dequant(A, state):
    return A.float() * (state / 127.0)


def int8_linear_matmul(A, B, out=None, dtype=torch.int32):
    raise NotImplementedError(
        "bitsandbytes shim: int8 matmul is not available on this stack "
        "(RDNA1 has no dp4a/tensor cores); use zluda_ops.quant int4/int8 "
        "dequant-once paths instead")


def int8_mm_dequant(A, row_stats, col_stats, bias=None, dtype=torch.float16):
    raise NotImplementedError("bitsandbytes shim: int8 mm dequant is not "
                              "available; use zluda_ops.quant.int4_matmul paths")
