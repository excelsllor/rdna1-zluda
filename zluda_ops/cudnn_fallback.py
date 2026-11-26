"""Automatic fallbacks for cuDNN operations ZLUDA does not map.

Status on this stack (2026-11-26):
- cuDNN batchnorm is unimplemented in zluda_dnn8 -> NOT_SUPPORTED. The native
  PyTorch path works, so BatchNorm forwards are wrapped and always run with
  cuDNN disabled on ZLUDA (installed by `install()`, autoloaded by
  `zluda_ops.auto`).
- cuDNN RNN (LSTM/GRU) is unimplemented and a failed cuDNN RNN call can leave
  MIOpen state unusable, so wrapping the call is not enough: the backend choice
  must happen with cuDNN disabled from the start. Use `rnn_context()` around
  model creation AND forward, or set `torch.backends.cudnn.enabled = False`
  globally for RNN models.

    from zluda_ops.cudnn_fallback import install, rnn_context
    install()
    with rnn_context():
        model = torch.nn.LSTM(...).cuda()
        y = model(x)
"""

import contextlib
import os

import torch

_installed = False
_orig_bn_forward = None


def _is_zluda():
    try:
        return torch.cuda.is_available() and "ZLUDA" in torch.cuda.get_device_name(0)
    except Exception:  # noqa: BLE001
        return False


_ZLUDA = _is_zluda()


def _bn_forward(self, input):
    if _ZLUDA:
        with torch.backends.cudnn.flags(enabled=False):
            return _orig_bn_forward(self, input)
    return _orig_bn_forward(self, input)


def install():
    """Patch BatchNorm forwards to run without cuDNN on ZLUDA."""
    global _installed, _orig_bn_forward
    if _installed:
        return False
    from torch.nn.modules.batchnorm import _BatchNorm

    _orig_bn_forward = _BatchNorm.forward
    _BatchNorm.forward = _bn_forward
    _installed = True
    return True


def uninstall():
    global _installed
    if not _installed:
        return False
    from torch.nn.modules.batchnorm import _BatchNorm

    _BatchNorm.forward = _orig_bn_forward
    _installed = False
    return True


def installed():
    return _installed


@contextlib.contextmanager
def rnn_context():
    """Context where cuDNN is disabled (create and run RNN models inside it)."""
    prev = torch.backends.cudnn.enabled
    torch.backends.cudnn.enabled = False
    try:
        yield
    finally:
        torch.backends.cudnn.enabled = prev


def autoload():
    if os.environ.get("ZLUDA_CUDNN_FALLBACK", "1") != "0":
        install()


__all__ = ["install", "uninstall", "installed", "autoload", "rnn_context"]