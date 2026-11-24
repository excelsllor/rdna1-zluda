import sys
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


print("torch", torch.__version__, "| device:", torch.cuda.get_device_name(0))
print("allow_tf32(matmul):", torch.backends.cuda.matmul.allow_tf32)

for dtype in (torch.float32, torch.float16, torch.bfloat16):
    for n in (512, 1024, 2048, 4096):
        try:
            a = torch.randn(n, n, device=dev, dtype=dtype)
            b = torch.randn(n, n, device=dev, dtype=dtype)
            ms = bench(lambda: a @ b)
            tflops = 2 * n**3 / (ms / 1000) / 1e12
            print(f"matmul {str(dtype).split('.')[-1]:9s} n={n:5d}: {ms:9.2f} ms  {tflops:7.2f} TFLOPS")
        except Exception as e:
            print(f"matmul {str(dtype).split('.')[-1]:9s} n={n:5d}: FAIL {type(e).__name__}: {str(e)[:150]}")

try:
    lin = torch.nn.Linear(1024, 1024).half().cuda()
    x = torch.randn(64, 1024, device=dev, dtype=torch.float16)
    ms = bench(lambda: lin(x))
    print(f"linear fp16 64x1024x1024: {ms:.2f} ms")
except Exception as e:
    print(f"linear fp16: FAIL {type(e).__name__}: {str(e)[:150]}")

try:
    import torch.nn.functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel

    backends = {
        "math": SDPBackend.MATH,
        "mem_efficient": SDPBackend.EFFICIENT_ATTENTION,
        "flash": SDPBackend.FLASH_ATTENTION,
    }
    for s in (512, 1024, 2048):
        for name, backend in backends.items():
            try:
                q = torch.randn(1, 8, s, 64, device=dev, dtype=torch.float16)
                k = torch.randn_like(q)
                v = torch.randn_like(q)
                with sdpa_kernel(backend):
                    f = lambda: F.scaled_dot_product_attention(q, k, v)
                    f()
                    torch.cuda.synchronize()
                    ms = bench(f, 3, 10)
                print(f"sdpa {name:13s} s={s}: {ms:8.2f} ms")
            except Exception as e:
                print(f"sdpa {name:13s} s={s}: FAIL {type(e).__name__}: {str(e)[:150]}")
except Exception as e:
    print("sdpa: FAIL", type(e).__name__, str(e)[:200])

try:
    x = torch.randn(16 * 1024 * 1024, device=dev)
    ms = bench(lambda: torch.sin(x) + torch.cos(x))
    print(f"elementwise 16M fp32: {ms:.2f} ms")
except Exception as e:
    print("elementwise: FAIL", type(e).__name__, str(e)[:150])

try:
    x = torch.randn(64, 4096, device=dev, dtype=torch.float16)
    ms = bench(lambda: torch.softmax(x, dim=-1))
    print(f"softmax 64x4096 fp16: {ms:.2f} ms")
except Exception as e:
    print("softmax: FAIL", type(e).__name__, str(e)[:150])
