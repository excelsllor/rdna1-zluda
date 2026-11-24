import os
import time

import torch

torch.manual_seed(0)
print("torch", torch.__version__, "| DISABLE_ADDMM_CUDA_LT =", os.environ.get("DISABLE_ADDMM_CUDA_LT"))


def bench(fn, warmup=5, iters=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000.0


for dtype in (torch.float16, torch.float32, torch.bfloat16):
    try:
        lin = torch.nn.Linear(1024, 1024, bias=True).to("cuda", dtype=dtype)
        x = torch.randn(64, 1024, device="cuda", dtype=dtype)
        y = lin(x)
        ref = (x.cpu().double() @ lin.weight.cpu().double().t() + lin.bias.cpu().double()).to(dtype)
        err = (y.cpu().double() - ref.double()).abs().max().item()
        ms = bench(lambda: lin(x))
        print(f"linear+bias {str(dtype).split('.')[-1]:9s}: {ms:7.2f} ms  max err {err:.5f}")
    except Exception as e:
        print(f"linear+bias {str(dtype).split('.')[-1]:9s}: FAIL {type(e).__name__}: {str(e)[:200]}")

# addmm without bias epilogue path (mm + explicit bias) also exercises cublasLt for fp16? just check correctness
try:
    a = torch.randn(64, 1024, device="cuda", dtype=torch.float16)
    b = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
    c = torch.randn(1024, device="cuda", dtype=torch.float16)
    out = torch.addmm(c, a, b)
    ref = (a.cpu().double() @ b.cpu().double() + c.cpu().double()).float()
    print("addmm fp16 max err:", (out.cpu() - ref).abs().max().item())
except Exception as e:
    print("addmm FAIL:", type(e).__name__, str(e)[:200])
