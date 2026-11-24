import torch, triton, triton.language as tl

@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, K, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK + tl.arange(0, BLOCK)
    offs_n = tl.arange(0, BLOCK)
    offs_k = tl.arange(0, BLOCK)
    acc = tl.zeros((BLOCK, BLOCK), dtype=tl.float32)
    for k in range(0, K, BLOCK):
        a = tl.load(a_ptr + offs_m[:, None] * K + (k + offs_k)[None, :])
        b = tl.load(b_ptr + (k + offs_k)[:, None] * N + offs_n[None, :])
        acc += tl.dot(a, b)
    tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], acc)

m = k = n = 256
a = torch.randn(m, k, device="cuda"); b = torch.randn(k, n, device="cuda"); c = torch.empty(m, n, device="cuda")
matmul_kernel[(m // 32,)](a, b, c, k, n, BLOCK=32)
torch.cuda.synchronize()
print("max err", (c - a @ b).abs().max().item())
