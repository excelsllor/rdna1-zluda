import torch
import triton
import triton.language as tl


@triton.jit
def one_dot(a_ptr, b_ptr, c_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK + tl.arange(0, BLOCK)
    offs_n = tl.arange(0, BLOCK)
    offs_k = tl.arange(0, BLOCK)
    a = tl.load(a_ptr + offs_m[:, None] * N + offs_k[None, :])
    b = tl.load(b_ptr + offs_k[:, None] * N + offs_n[None, :])
    acc = tl.dot(a, b)
    tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], acc.to(tl.float16))


n = 64
torch.manual_seed(1)
a = torch.randn(n, 32, device="cuda", dtype=torch.float16)
b = torch.randn(32, n, device="cuda", dtype=torch.float16)
c = torch.empty(n, n, device="cuda", dtype=torch.float16)
one_dot[(2,)](a, b, c, n, BLOCK=32)
torch.cuda.synchronize()
ref = (a.cpu().double() @ b.cpu().double()).float()
err = (c.cpu() - ref).abs()
block0 = err[:32]
block1 = err[32:]
print("block0 max err:", block0.max().item())
print("block1 max err:", block1.max().item())
print("block0 err per row (max):", [round(v, 3) for v in block0.max(dim=1).values.tolist()])
print("block1 err per row (max):", [round(v, 3) for v in block1.max(dim=1).values.tolist()])
print("c[32,:4]  :", c[32, :4].tolist())
print("ref[32,:4]:", ref[32, :4].tolist())
print("c[32,28:32]  :", c[32, 28:].tolist())
print("ref[32,28:32]:", ref[32, 28:].tolist())
