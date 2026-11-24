import torch

n = 256
torch.manual_seed(0)

for dtype in (torch.float32, torch.float16, torch.bfloat16):
    a = torch.randn(n, n, device="cuda", dtype=dtype)
    b = torch.randn(n, n, device="cuda", dtype=dtype)
    c = a @ b
    ref = a.cpu().double() @ b.cpu().double()
    err = (c.cpu().double() - ref).abs().max().item()
    scale = ref.abs().max().item()
    print(f"torch {str(dtype):15s} max err {err:.5f}  (scale {scale:.1f})")

# triton fp16 dot (mma path)
import triton
import triton.language as tl


@triton.jit
def mm(a_ptr, b_ptr, c_ptr, K, N, BLOCK: tl.constexpr):
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


a = torch.randn(n, n, device="cuda", dtype=torch.float16)
b = torch.randn(n, n, device="cuda", dtype=torch.float16)
c = torch.empty(n, n, device="cuda", dtype=torch.float16)
mm[(n // 32,)](a, b, c, n, n, BLOCK=32)
torch.cuda.synchronize()
ref = (a.cpu().double() @ b.cpu().double()).float()
print("triton fp16 max err:", (c.cpu() - ref).abs().max().item())

# single dot, no K loop
@triton.jit
def mm1(a_ptr, b_ptr, c_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK + tl.arange(0, BLOCK)
    offs_n = tl.arange(0, BLOCK)
    offs_k = tl.arange(0, BLOCK)
    a = tl.load(a_ptr + offs_m[:, None] * BLOCK + offs_k[None, :])
    b = tl.load(b_ptr + offs_k[:, None] * N + offs_n[None, :])
    acc = tl.dot(a, b)
    tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], acc.to(tl.float16))


a = torch.randn(32, 32, device="cuda", dtype=torch.float16)
b = torch.randn(32, 32, device="cuda", dtype=torch.float16)
c = torch.empty(32, 32, device="cuda", dtype=torch.float16)
mm1[(1,)](a, b, c, 32, BLOCK=32)
torch.cuda.synchronize()
ref = (a.cpu().double() @ b.cpu().double()).float()
print("triton single dot max err:", (c.cpu() - ref).abs().max().item())
print("first row c  :", c[0, :4].tolist())
print("first row ref:", ref[0, :4].tolist())
