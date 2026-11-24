import sys

import torch

print("torch", torch.__version__, "cuda", torch.version.cuda)
print("available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    sys.exit(1)
print("device:", torch.cuda.get_device_name(0))
props = torch.cuda.get_device_properties(0)
print("props:", props)

for dtype in (torch.float32, torch.float16, torch.bfloat16):
    try:
        a = torch.randn(512, 512, device="cuda", dtype=dtype)
        b = torch.randn(512, 512, device="cuda", dtype=dtype)
        c = a @ b
        torch.cuda.synchronize()
        print(dtype, "matmul OK", tuple(c.shape), "mean", float(c.abs().mean()))
    except Exception as e:
        print(dtype, "matmul FAILED:", type(e).__name__, str(e)[:300])

try:
    x = torch.randn(64, 128, 256, device="cuda")
    w = torch.randn(512, 256, device="cuda")
    y = torch.nn.functional.gelu(x @ w.t())
    torch.cuda.synchronize()
    print("mlp OK", tuple(y.shape))
except Exception as e:
    print("mlp FAILED:", type(e).__name__, str(e)[:300])
