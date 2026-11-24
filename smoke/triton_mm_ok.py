import torch
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, BLOCK: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK + tl.arange(0, BLOCK)
    offs_n = pid_n * BLOCK + tl.arange(0, BLOCK)
    offs_k = tl.arange(0, BLOCK)
    acc = tl.zeros((BLOCK, BLOCK), dtype=tl.float32)
    for k in range(0, K, BLOCK):
        a = tl.load(a_ptr + offs_m[:, None] * K + (k + offs_k)[None, :])
        b = tl.load(b_ptr + (k + offs_k)[:, None] * N + offs_n[None, :])
        acc += tl.dot(a, b)
    tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], acc.to(tl.float16))


def check(n, block=64):
    a = torch.randn(n, n, device="cuda", dtype=torch.float16)
    b = torch.randn(n, n, device="cuda", dtype=torch.float16)
    c = torch.empty(n, n, device="cuda", dtype=torch.float16)
    grid = (n // block, n // block)
    matmul_kernel[grid](a, b, c, n, n, n, BLOCK=block)
    torch.cuda.synchronize()
    ref = (a.cpu().double() @ b.cpu().double()).float()
    err = (c.cpu() - ref).abs().max().item()
    print(f"triton matmul n={n} block={block}: max err {err:.4f}")
    return err


for n in (64, 128, 256, 512):
    check(n)
