import torch
from torch import nn

_orig_forward = None


def _slow_forward(self, *args, **kwargs):
    if "need_weights" in kwargs:
        need_weights = kwargs["need_weights"]
        kwargs = dict(kwargs)
        kwargs["need_weights"] = True
    elif len(args) >= 5:
        need_weights = args[4]
        args = list(args)
        args[4] = True
    else:
        need_weights = True
    torch.backends.mha.set_fastpath_enabled(False)
    out, weights = _orig_forward(self, *args, **kwargs)
    return (out, weights) if need_weights else out


try:
    from torch._dynamo import disable as _dynamo_disable
    _slow_forward = _dynamo_disable(_slow_forward)
except Exception:
    pass


def install():
    global _orig_forward
    if _orig_forward is not None:
        return False
    _orig_forward = nn.MultiheadAttention.forward
    nn.MultiheadAttention.forward = _slow_forward
    return True


def uninstall():
    global _orig_forward
    if _orig_forward is None:
        return False
    nn.MultiheadAttention.forward = _orig_forward
    _orig_forward = None
    return True


def installed():
    return _orig_forward is not None


__all__ = ["install", "uninstall", "installed"]
