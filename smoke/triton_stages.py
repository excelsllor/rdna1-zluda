import torch
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


def run(n, stages, block=32):
    a = torch.randn(n, n, device="cuda", dtype=torch.float16)
    b = torch.randn(n, n, device="cuda", dtype=torch.float16)
    c = torch.empty(n, n, device="cuda", dtype=torch.float16)
    mm[(n // block,)](a, b, c, n, n, BLOCK=block, num_stages=stages)
    torch.cuda.synchronize()
    ref = (a.cpu().double() @ b.cpu().double()).float()
    err = (c.cpu() - ref).abs().max().item()
    print(f"n={n} stages={stages}: max err {err:.5f}")
    return err


for stages in (1, 2, 3):
    for n in (32, 64, 128, 256):
        try:
            run(n, stages)
        except Exception as e:
            print(f"n={n} stages={stages}: FAILED {type(e).__name__}: {str(e)[:120]}")
