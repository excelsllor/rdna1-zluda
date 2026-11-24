import time

import torch
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
from torch.nn.attention import SDPBackend, sdpa_kernel

torch.manual_seed(0)
B, H, S, D = 1, 8, 512, 64
q = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
k = torch.randn_like(q)
v = torch.randn_like(q)


def causal(b, h, qi, ki):
    return qi >= ki


try:
    bm = create_block_mask(causal, B=None, H=None, Q_LEN=S, KV_LEN=S, device="cuda")
    f = torch.compile(flex_attention, dynamic=False)
    out = f(q, k, v, block_mask=bm)
    torch.cuda.synchronize()
    with sdpa_kernel(SDPBackend.MATH):
        ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    err = (out.float() - ref.float()).abs().max().item()
    def bench(fn, warmup=3, iters=10):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / iters * 1000.0
    ms = bench(lambda: f(q, k, v, block_mask=bm))
    print(f"flex_attention OK: {ms:.2f} ms  max err {err:.4f}")
except Exception as e:
    print(f"flex_attention FAIL: {type(e).__name__}: {str(e)[:400]}")
