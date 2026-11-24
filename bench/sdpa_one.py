import sys
import time

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

name = sys.argv[1]
backend = {
    "math": SDPBackend.MATH,
    "mem_efficient": SDPBackend.EFFICIENT_ATTENTION,
    "flash": SDPBackend.FLASH_ATTENTION,
}[name]

torch.manual_seed(0)


def bench(fn, warmup=3, iters=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000.0


for s in (512, 1024, 2048):
    q = torch.randn(1, 8, s, 64, device="cuda", dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    with sdpa_kernel(backend):
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        torch.cuda.synchronize()
        ms = bench(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True))
    ref = F.scaled_dot_product_attention(q.cpu(), k.cpu(), v.cpu(), is_causal=True)
    err = (out.cpu().float() - ref.float()).abs().max().item()
    print(f"sdpa {name} s={s}: {ms:8.2f} ms  max err {err:.4f}")
