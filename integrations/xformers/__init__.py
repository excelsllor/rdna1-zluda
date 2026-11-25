"""xformers-compatible shim for the ops most code uses on gfx1010.

Real xformers cannot be installed/used here: its memory-efficient attention
kernels are CUDA/CUTLASS sm80+ and RDNA1 has no matrix cores. This shim maps
`xformers.ops.memory_efficient_attention` (and friends) to the ZLUDA Triton
flash-attention path from `zluda_ops`, with a math-SDPA fallback.

Supports: `memory_efficient_attention`, `memory_efficient_attention_forward`,
`AttentionBias`, `LowerTriangularMask`, `TensorBias`, `BlockDiagonalMask`
(minimal), and `xformers.ops.fmha` aliases. Layouts follow xformers: (B, S, H, D)
tensors and optional attn_bias; unsupported biases fall back to math SDPA.
"""

import torch
import torch.nn.functional as F

from . import ops  # noqa: F401

__version__ = "0.0.29+zluda-shim"

__all__ = ["ops", "__version__"]
