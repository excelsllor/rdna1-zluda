import time

import torch

torch.manual_seed(0)
dev = "cuda"
print("torch", torch.__version__, "| device:", torch.cuda.get_device_name(0))
print("allow_fp16_accumulation default:", torch._C._get_cublas_allow_fp16_accumulation())


def bench(fn, warmup=5, iters=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000.0


for acc in (False, True):
    torch._C._set_cublas_allow_fp16_accumulation(acc)
    print(f"--- fp16 accumulation = {acc} ---")
    for n in (1024, 2048, 4096):
        a = torch.randn(n, n, device=dev, dtype=torch.float16)
        b = torch.randn(n, n, device=dev, dtype=torch.float16)
        c = a @ b
        ref = (a.cpu().double() @ b.cpu().double()).float()
        err = (c.cpu() - ref).abs().max().item()
        ms = bench(lambda: a @ b)
        print(
            f"matmul fp16 n={n:5d}: {ms:8.2f} ms  {2*n**3/(ms/1000)/1e12:6.2f} TFLOPS  max err {err:.4f}"
        )
    lin = torch.nn.Linear(1024, 1024, bias=True).half().cuda()
    x = torch.randn(256, 1024, device=dev, dtype=torch.float16)
    y = lin(x)
    ref = (x.cpu().double() @ lin.weight.cpu().double().t() + lin.bias.cpu().double()).float()
    err = (y.cpu() - ref).abs().max().item()
    ms = bench(lambda: lin(x))
    print(f"linear fp16 256x1024x1024: {ms:7.2f} ms  max err {err:.4f}")

torch._C._set_cublas_allow_fp16_accumulation(False)
