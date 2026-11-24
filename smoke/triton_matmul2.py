import torch
import triton
import triton.language as tl


@triton.jit
def matmul_f16(a_ptr, b_ptr, c_ptr, K, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK + tl.arange(0, BLOCK)
    offs_n = tl.arange(0, BLOCK)
    offs_k = tl.arange(0, BLOCK)
    acc = tl.zeros((BLOCK, BLOCK), dtype=tl.float32)
    for k in range(0, K, BLOCK):
        a = tl.load(a_ptr + offs_m[:, None] * K + (k + offs_k)[None, :])
        b = tl.load(b_ptr + (k + offs_k)[:, None] * N + offs_n[None, :])
        acc += tl.dot(a, b)
    tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], acc.to(tl.float16))


@triton.jit
def matmul_f32_ieee(a_ptr, b_ptr, c_ptr, K, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK + tl.arange(0, BLOCK)
    offs_n = tl.arange(0, BLOCK)
    offs_k = tl.arange(0, BLOCK)
    acc = tl.zeros((BLOCK, BLOCK), dtype=tl.float32)
    for k in range(0, K, BLOCK):
        a = tl.load(a_ptr + offs_m[:, None] * K + (k + offs_k)[None, :])
        b = tl.load(b_ptr + (k + offs_k)[:, None] * N + offs_n[None, :])
        acc += tl.dot(a, b, input_precision="ieee")
    tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], acc)


def run(name, kernel, dtype, n=256, block=32):
    a = torch.randn(n, n, device="cuda", dtype=dtype)
    b = torch.randn(n, n, device="cuda", dtype=dtype)
    c = torch.empty(n, n, device="cuda", dtype=dtype)
    kernel[(n // block,)](a, b, c, n, n, BLOCK=block)
    torch.cuda.synchronize()
    ref = a @ b
    err = (c.float() - ref.float()).abs().max().item()
    print(f"{name}: max err {err:.6f}")


for name, kern, dt in (
    ("fp16 tl.dot", matmul_f16, torch.float16),
    ("fp32 tl.dot ieee", matmul_f32_ieee, torch.float32),
):
    try:
        run(name, kern, dt)
    except Exception as e:
        print(f"{name}: FAILED {type(e).__name__}: {str(e)[:200]}")
