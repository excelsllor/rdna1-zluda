"""bitsandbytes.optim shim: 8-bit optimizers mapped to our Triton kernels."""

from zluda_ops.optim8bit import AdamW8bit as _AdamW8bit


class AdamW8bit(_AdamW8bit):
    def __init__(self, params, lr=1e-2, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.01, **kwargs):
        super().__init__(params, lr=lr, betas=betas, eps=eps,
                         weight_decay=weight_decay)


class Adam8bit(AdamW8bit):
    pass


class GlobalOptimManager:
    """Minimal stub for code that registers it (no-op here)."""

    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance


def __getattr__(name):
    raise AttributeError(
        f"bitsandbytes.optim shim: {name} is not implemented "
        f"(only AdamW8bit/Adam8bit are available)")


__all__ = ["AdamW8bit", "Adam8bit", "GlobalOptimManager"]
