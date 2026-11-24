import time

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


def bench(fn, warmup=3, iters=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000.0


torch.manual_seed(0)
for s in (512, 1024, 2048):
    q = torch.randn(1, 8, s, 64, device="cuda", dtype=torch.float16, requires_grad=True)
    k = torch.randn(1, 8, s, 64, device="cuda", dtype=torch.float16, requires_grad=True)
    v = torch.randn(1, 8, s, 64, device="cuda", dtype=torch.float16, requires_grad=True)
    do = torch.randn_like(q)
    with sdpa_kernel(SDPBackend.MATH):
        ms_fwd = bench(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True))
        def fb():
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            o.backward(do)
            q.grad = k.grad = v.grad = None
        ms_fb = bench(fb)
    print(f"math s={s}: fwd {ms_fwd:.2f} ms | fwd+bwd {ms_fb:.2f} ms")
