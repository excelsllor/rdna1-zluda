"""bitsandbytes-compatible shim for the ZLUDA/RDNA1 stack.

The real bitsandbytes kernels cannot run under ZLUDA (nvcc-built SASS without
matching ISA, and the ROCm build needs a HIP toolchain), so this shim maps the
optimizer API to `zluda_ops.optim8bit.AdamW8bit` (Triton blockwise-int8 states)
and provides clear errors for the quantization APIs that have no equivalent.

Install with `tools/install_ecosystem.py` (adds integrations/ to sys.path):

    import bitsandbytes as bnb
    opt = bnb.optim.AdamW8bit(model.parameters(), lr=1e-3)
"""

from . import functional, nn, optim  # noqa: F401

__version__ = "0.50.2+zluda-shim"

__all__ = ["functional", "nn", "optim", "__version__"]
