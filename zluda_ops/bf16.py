import torch
import torch.nn.functional as F


def matmul(a, b):
    if a.dtype == torch.bfloat16 and b.dtype == torch.bfloat16 and a.is_cuda:
        return (a.float() @ b.float()).to(torch.bfloat16)
    return a @ b


def linear(x, weight, bias=None):
    if x.dtype == torch.bfloat16 and weight.dtype == torch.bfloat16 and x.is_cuda:
        y = (x.float() @ weight.float().t()).to(torch.bfloat16)
        if bias is not None:
            y = (y.float() + bias.float()).to(torch.bfloat16)
        return y
    return F.linear(x, weight, bias)


__all__ = ["matmul", "linear"]
