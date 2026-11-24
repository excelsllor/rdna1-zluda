import time

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
dev = "cuda"
B, S, H, HD = 8, 256, 8, 64
D = H * HD


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(D)
        self.qkv = nn.Linear(D, 3 * D)
        self.proj = nn.Linear(D, D)
        self.ln2 = nn.LayerNorm(D)
        self.fc1 = nn.Linear(D, 4 * D)
        self.fc2 = nn.Linear(4 * D, D)

    def forward(self, x):
        h = self.ln1(x)
        qkv = self.qkv(h).view(B, S, 3, H, HD).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        att = (q @ k.transpose(-2, -1)) * (HD**-0.5)
        mask = torch.triu(torch.ones(S, S, device=x.device, dtype=torch.bool), diagonal=1)
        att = att.masked_fill(mask, float("-inf"))
        att = torch.softmax(att, dim=-1)
        o = (att @ v).transpose(1, 2).reshape(B, S, D)
        h = x + self.proj(o)
        h = h + self.fc2(F.gelu(self.fc1(self.ln2(h))))
        return h


def bench(fn, warmup=3, iters=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000.0


model = Block().half().cuda()
x = torch.randn(B, S, D, device=dev, dtype=torch.float16)

with torch.no_grad():
    ms_eager = bench(lambda: model(x))
    print(f"eager fp16: {ms_eager:.2f} ms")

    print("compiling...")
    t0 = time.time()
    compiled = torch.compile(model)
    y = compiled(x)
    torch.cuda.synchronize()
    print(f"compile+first run: {time.time() - t0:.1f} s")
    ms_comp = bench(lambda: compiled(x))
    print(f"compiled fp16: {ms_comp:.2f} ms")
    err = (y - model(x)).abs().max().item()
    print(f"max diff vs eager: {err:.4f}")
