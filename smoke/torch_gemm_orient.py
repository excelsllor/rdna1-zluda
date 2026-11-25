import time

import torch

torch.manual_seed(0)
a = torch.randn(4096, 4096, device="cuda", dtype=torch.float32)
b = torch.randn(4096, 4096, device="cuda", dtype=torch.float32)

for _ in range(3):
    c = a @ b
torch.cuda.synchronize()
t = time.perf_counter()
for _ in range(5):
    c = a @ b
torch.cuda.synchronize()
ms = (time.perf_counter() - t) / 5 * 1000
print("a@b", ms, 2 * 4096**3 / (ms / 1000) / 1e12, flush=True)

for label, fn in (("bT@aT", lambda: b.t() @ a.t()), ("a@bT", lambda: a @ b.t()), ("aT@b", lambda: a.t() @ b)):
    try:
        fn()
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        ms2 = (time.perf_counter() - t) / 5 * 1000
        print(label, ms2, 2 * 4096**3 / (ms2 / 1000) / 1e12, flush=True)
    except Exception as exc:
        print(label, "ERR", exc, flush=True)
