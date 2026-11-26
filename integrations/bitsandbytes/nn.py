"""bitsandbytes.nn shim: modules that need real bnb kernels raise clearly."""

import torch

__all__ = ["Linear8bitLt", "Linear4bit"]


class Linear8bitLt(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "bitsandbytes shim: Linear8bitLt needs the bnb CUDA kernels, which "
            "cannot run under ZLUDA. Use zluda_ops.quant.quantize_int4/8 + "
            "int4_linear/int4_matmul (fused dequant kernels) instead.")


class Linear4bit(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "bitsandbytes shim: Linear4bit needs the bnb CUDA kernels, which "
            "cannot run under ZLUDA. Use zluda_ops.quant.int4_linear / "
            "int4_matmul instead.")
