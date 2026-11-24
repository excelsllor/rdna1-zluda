import torch
import triton
import triton.language as tl


@triton.jit
def dot_trans(a_ptr, b_ptr, c_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
              BM: tl.constexpr, BN: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, K)
    a = tl.load(a_ptr + offs_m[:, None] * K + offs_k[None, :])
    b = tl.load(b_ptr + offs_n[:, None] * K + offs_k[None, :])  # [BN, K]
    c = tl.dot(a, tl.trans(b))
    tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], c)


torch.manual_seed(0)
for (m, n, k, bm, bn, warps) in [
    (32, 32, 32, 32, 32, 4),
    (64, 64, 32, 32, 32, 4),
    (32, 64, 32, 32, 32, 4),
    (64, 32, 32, 32, 32, 4),
    (64, 64, 64, 64, 64, 4),
]:
    a = torch.randn(m, k, device="cuda", dtype=torch.float16)
    b = torch.randn(n, k, device="cuda", dtype=torch.float16)
    c = torch.empty(m, n, device="cuda", dtype=torch.float16)
    try:
        dot_trans[(m // bm, n // bn)](a, b, c, m, n, k, bm, bn, num_warps=warps, num_stages=1)
        torch.cuda.synchronize()
        ref = a @ b.t()
        err = (c.float() - ref.float()).abs().max().item()
        print(f"m={m} n={n} k={k} bm={bm} bn={bn} warps={warps}: max err {err:.5f}")
    except Exception as e:
        print(f"m={m} n={n} k={k}: FAIL {type(e).__name__}: {str(e)[:200]}")
