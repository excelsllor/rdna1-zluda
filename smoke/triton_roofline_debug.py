import traceback
import torch
import triton
import triton.language as tl

@triton.jit
def k_add(out_ptr, ITER: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    acc = tl.load(out_ptr + offs)
    for _ in range(ITER):
        acc = acc + 1.0
    tl.store(out_ptr + offs, acc)

@triton.jit
def k_fma_scalar(out_ptr, a, b, ITER: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    acc = tl.load(out_ptr + offs)
    for _ in range(ITER):
        acc = acc * a + b
    tl.store(out_ptr + offs, acc)

@triton.jit
def k_fma_ptr(out_ptr, a_ptr, b_ptr, ITER: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    acc = tl.load(out_ptr + offs)
    a = tl.load(a_ptr)
    b = tl.load(b_ptr)
    for _ in range(ITER):
        acc = acc * a + b
    tl.store(out_ptr + offs, acc)

@triton.jit
def k_chase(next_ptr, out_ptr, N: tl.constexpr):
    idx = tl.zeros((1,), tl.int32)
    for _ in range(N):
        idx = tl.load(next_ptr + idx)
    tl.store(out_ptr + tl.arange(0, 1), idx)

def run(name, fn, sync=True):
    try:
        fn()
        if sync:
            torch.cuda.synchronize()
        print(name, "OK")
    except Exception as e:
        print(name, "FAIL", type(e).__name__, str(e)[:200])
        traceback.print_exc(limit=2)

torch.manual_seed(0)
n = 256 * 8
out = torch.randn(n, device="cuda", dtype=torch.float32)
a = torch.tensor(1.0001, device="cuda", dtype=torch.float32)
b = torch.tensor(1e-5, device="cuda", dtype=torch.float32)
run("add iter=4", lambda: k_add[(8,)](out, ITER=4, BLOCK=256, num_warps=4))
run("fma scalar iter=4", lambda: k_fma_scalar[(8,)](out, 1.0001, 1e-5, ITER=4, BLOCK=256, num_warps=4))
run("fma scalar iter=2048", lambda: k_fma_scalar[(8,)](out, 1.0001, 1e-5, ITER=2048, BLOCK=256, num_warps=4))
run("fma ptr iter=4", lambda: k_fma_ptr[(8,)](out, a, b, ITER=4, BLOCK=256, num_warps=4))
run("fma ptr iter=2048", lambda: k_fma_ptr[(8,)](out, a, b, ITER=2048, BLOCK=256, num_warps=4))
nxt = torch.randint(0, 1<<20, (1<<20,), device="cuda", dtype=torch.int32)
res = torch.zeros(1, device="cuda", dtype=torch.int32)
run("chase 8", lambda: k_chase[(1,)](nxt, res, N=8, num_warps=1))
run("chase 512", lambda: k_chase[(1,)](nxt, res, N=512, num_warps=1))
