import torch

torch.manual_seed(0)

for dtype in (torch.float32, torch.float16):
    x = torch.randn(64, 1024, device="cuda", dtype=dtype)
    W = torch.randn(1024, 1024, device="cuda", dtype=dtype)
    b = torch.randn(1024, device="cuda", dtype=dtype)

    y_mm = x @ W.t()  # cublas path (known good)
    y_addmm = torch.addmm(b, x, W.t())  # cublasLt path with bias epilogue
    y_sep = y_mm + b

    ref = (x.cpu().double() @ W.cpu().double().t()).to(dtype) + b.cpu().double().to(dtype)
    print(
        f"{str(dtype):15s} err(mm)= {(y_mm.cpu().double() - ref.double()).abs().max().item():.5f}"
        f" err(addmm)= {(y_addmm.cpu().double() - ref.double()).abs().max().item():.5f}"
        f" err(sep)= {(y_sep.cpu().double() - ref.double()).abs().max().item():.5f}"
        f" |addmm-sep|= {(y_addmm - y_sep).abs().max().item():.5f}"
    )

# bias-only check: zero matrix product, bias must appear
try:
    x0 = torch.zeros(8, 16, device="cuda", dtype=torch.float32)
    W0 = torch.zeros(16, 32, device="cuda", dtype=torch.float32)
    b0 = torch.arange(32, device="cuda", dtype=torch.float32)
    y0 = torch.addmm(b0, x0, W0)
    print("bias-only addmm row0:", y0[0, :8].tolist())
except Exception as e:
    print("bias-only FAIL", type(e).__name__, str(e)[:150])
