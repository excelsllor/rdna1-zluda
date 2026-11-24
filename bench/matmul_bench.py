import time

import torch

dev = "cuda"
torch.manual_seed(0)


def bench(fn, warmup=5, iters=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000.0


print("torch", torch.__version__)
n = 512
for dtype in (torch.float32, torch.float16):
    a = torch.randn(n, n, device=dev, dtype=dtype)
    b = torch.randn(n, n, device=dev, dtype=dtype)
    c = a @ b
    ref = (a.cpu().double() @ b.cpu().double()).to(dtype)
    err = (c.cpu().double() - ref.double()).abs().max().item()
    print(f"correctness {str(dtype):15s}: max err {err:.5f}")

for dtype in (torch.float32, torch.float16, torch.bfloat16):
    for n in (1024, 2048, 4096):
        a = torch.randn(n, n, device=dev, dtype=dtype)
        b = torch.randn(n, n, device=dev, dtype=dtype)
        ms = bench(lambda: a @ b)
        print(f"matmul {str(dtype).split('.')[-1]:9s} n={n:5d}: {ms:9.2f} ms  {2*n**3/(ms/1000)/1e12:7.2f} TFLOPS")
