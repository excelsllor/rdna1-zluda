import time

import torch

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
print("DISABLE_ADDMM_CUDA_LT =", __import__("os").environ.get("DISABLE_ADDMM_CUDA_LT"))

# linear / addmm with bias
for dtype in (torch.float16, torch.float32, torch.bfloat16):
    try:
        lin = torch.nn.Linear(1024, 1024).to(dev := "cuda", dtype=dtype)
        x = torch.randn(64, 1024, device=dev, dtype=dtype)
        y = lin(x)
        ref = (x.cpu().double() @ lin.weight.cpu().double().t() + lin.bias.cpu().double()).to(dtype)
        err = (y.cpu().double() - ref.double()).abs().max().item()
        ms = bench(lambda: lin(x))
        print(f"linear {str(dtype).split('.')[-1]:9s} 64x1024x1024: {ms:7.2f} ms  max err {err:.4f}")
    except Exception as e:
        print(f"linear {str(dtype).split('.')[-1]:9s}: FAIL {type(e).__name__}: {str(e)[:150]}")

# elementwise
try:
    x = torch.randn(16 * 1024 * 1024, device="cuda")
    ms = bench(lambda: torch.sin(x) + torch.cos(x))
    print(f"elementwise 16M fp32: {ms:.2f} ms")
except Exception as e:
    print("elementwise: FAIL", type(e).__name__, str(e)[:150])

# softmax / layernorm / gelu
try:
    x = torch.randn(64, 4096, device="cuda", dtype=torch.float16)
    ms = bench(lambda: torch.softmax(x, dim=-1))
    print(f"softmax 64x4096 fp16: {ms:.2f} ms")
except Exception as e:
    print("softmax: FAIL", type(e).__name__, str(e)[:150])

try:
    ln = torch.nn.LayerNorm(1024).half().cuda()
    x = torch.randn(64, 1024, device="cuda", dtype=torch.float16)
    ms = bench(lambda: ln(x))
    print(f"layernorm 64x1024 fp16: {ms:.2f} ms")
except Exception as e:
    print("layernorm: FAIL", type(e).__name__, str(e)[:150])

try:
    import torch.nn.functional as F
    x = torch.randn(64, 4096, device="cuda", dtype=torch.float16)
    ms = bench(lambda: F.gelu(x))
    print(f"gelu 64x4096 fp16: {ms:.2f} ms")
except Exception as e:
    print("gelu: FAIL", type(e).__name__, str(e)[:150])

# SDPA math only (isolated from other backends)
try:
    import torch.nn.functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel

    for s in (512, 1024, 2048):
        q = torch.randn(1, 8, s, 64, device="cuda", dtype=torch.float16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        with sdpa_kernel(SDPBackend.MATH):
            f = lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True)
            f()
            torch.cuda.synchronize()
            ms = bench(f, 3, 10)
        print(f"sdpa math causal s={s}: {ms:8.2f} ms")
except Exception as e:
    print("sdpa math: FAIL", type(e).__name__, str(e)[:200])
