"""Model-side patches: fused kernels in HF transformer forwards (ZLUDA/RDNA1).

Currently patches Qwen2-style MLP so that single-token decode uses the fused
`sila(x@Wg + bg) * (x@Wu + bu)` GEMV (`zluda_ops.fused.swiglu_gemv`) instead of
two separate projections plus an elementwise kernel. Other paths are untouched.

Usage:
    from zluda_ops import model_patches
    model_patches.install()   # optional; GraphDecoder users benefit most
    model_patches.uninstall()
"""

import torch

_orig = {}
_installed = False


def _mlp_forward(self, x):
    from . import fused as _fused

    if (x.dim() == 3 and x.shape[0] == 1 and x.shape[1] <= 16
            and x.dtype == torch.float16 and x.is_contiguous()):
        flat = x.reshape(-1, x.shape[-1])
        out = _fused.swiglu_gemv(flat, self.gate_proj.weight,
                                 self.up_proj.weight, self.gate_proj.bias,
                                 self.up_proj.bias)
        if out is not None:
            return self.down_proj(out.reshape(x.shape[:-1] + (out.shape[0],)))
    gate = self.gate_proj(x)
    up = self.up_proj(x)
    return self.down_proj(self.act_fn(gate) * up)


def install():
    """Patch Qwen2 MLP forward with the fused decode path."""
    global _installed
    if _installed:
        return False
    from transformers.models.qwen2 import modeling_qwen2 as mq

    _orig["mlp"] = mq.Qwen2MLP.forward
    mq.Qwen2MLP.forward = _mlp_forward
    _installed = True
    return True


def uninstall():
    global _installed
    if not _installed:
        return False
    from transformers.models.qwen2 import modeling_qwen2 as mq

    mq.Qwen2MLP.forward = _orig["mlp"]
    _installed = False
    return True


def installed():
    return _installed


__all__ = ["install", "uninstall", "installed"]