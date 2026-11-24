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


@triton.jit
def two_dot_loop(a_ptr, b_ptr, c_ptr, N, BLOCK: tl.constexpr):
    offs_m = tl.arange(0, BLOCK)
    offs_n = tl.arange(0, BLOCK)
    offs_k = tl.arange(0, BLOCK)
    acc = tl.zeros((BLOCK, BLOCK), dtype=tl.float32)
    for k in range(0, 2 * BLOCK, BLOCK):
        a = tl.load(a_ptr + offs_m[:, None] * N + (k + offs_k)[None, :])
        b = tl.load(b_ptr + (k + offs_k)[:, None] * N + offs_n[None, :])
        acc += tl.dot(a, b)
    tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], acc.to(tl.float16))


@triton.jit
def two_dot_unrolled(a_ptr, b_ptr, c_ptr, N, BLOCK: tl.constexpr):
    offs_m = tl.arange(0, BLOCK)
    offs_n = tl.arange(0, BLOCK)
    offs_k = tl.arange(0, BLOCK)
    a0 = tl.load(a_ptr + offs_m[:, None] * N + offs_k[None, :])
    b0 = tl.load(b_ptr + offs_k[:, None] * N + offs_n[None, :])
    acc = tl.dot(a0, b0)
    a1 = tl.load(a_ptr + offs_m[:, None] * N + (BLOCK + offs_k)[None, :])
    b1 = tl.load(b_ptr + (BLOCK + offs_k)[:, None] * N + offs_n[None, :])
    acc += tl.dot(a1, b1)
    tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], acc.to(tl.float16))


def ref_of(a, b):
    return (a.cpu().double() @ b.cpu().double()).float()


def check(name, fn, n, grid, kdim=None):
    k = kdim or n
    a = torch.randn(n, k, device="cuda", dtype=torch.float16)
    b = torch.randn(k, n, device="cuda", dtype=torch.float16)
    c = torch.empty(n, n, device="cuda", dtype=torch.float16)
    fn[(grid,)](a, b, c, n, BLOCK=32)
    torch.cuda.synchronize()
    err = (c.cpu() - ref_of(a, b)).abs().max().item()
    print(f"{name}: max err {err:.5f}")


check("one_dot grid=1 (32x32)", one_dot, 32, 1, kdim=32)
check("one_dot grid=2 (64 rows)", one_dot, 64, 2, kdim=32)
check("one_dot grid=4 (128 rows)", one_dot, 128, 4, kdim=32)
check("two_dot_unrolled (32x64 @ 64x32)", two_dot_unrolled, 32, 1, kdim=64)
check("two_dot_loop (32x64 @ 64x32)", two_dot_loop, 32, 1, kdim=64)
