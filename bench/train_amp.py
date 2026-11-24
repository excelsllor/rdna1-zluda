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
        self.head = nn.Linear(D, 1)

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
        return self.head(h).mean()


x = torch.randn(B, S, D, device=dev, dtype=torch.float32)
y = torch.randn((), device=dev, dtype=torch.float32)


def run(model, steps, scaler=None, autocast_dtype=torch.float16):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    losses = []
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=autocast_dtype):
            out = model(x)
            loss = F.mse_loss(out.float(), y)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            opt.step()
        losses.append(loss.item())
    torch.cuda.synchronize()
    return losses


def timeit(fn, steps):
    t0 = time.perf_counter()
    losses = fn(steps)
    return (time.perf_counter() - t0) / steps * 1000, losses


model = Block().float().cuda()
scaler = torch.cuda.amp.GradScaler()
run(model, 3, scaler)
ms, losses = timeit(lambda n: run(model, n, scaler), 10)
print(f"eager AMP fp16: {ms:.2f} ms/step  loss {losses[0]:.4f} -> {losses[-1]:.4f} nan={any(l != l for l in losses)}")

model_bf = Block().float().cuda()
model_bf.load_state_dict(model.state_dict())
run(model_bf, 3, None, torch.bfloat16)
ms, losses = timeit(lambda n: run(model_bf, n, None, torch.bfloat16), 10)
print(f"eager AMP bf16: {ms:.2f} ms/step  loss {losses[0]:.4f} -> {losses[-1]:.4f} nan={any(l != l for l in losses)}")

try:
    model_c = Block().float().cuda()
    model_c.load_state_dict(model.state_dict())
    compiled = torch.compile(model_c)
    scaler2 = torch.cuda.amp.GradScaler()
    t0 = time.perf_counter()
    run(compiled, 3, scaler2)
    compile_time = time.perf_counter() - t0
    ms, losses = timeit(lambda n: run(compiled, n, scaler2), 10)
    print(f"compiled AMP fp16: {ms:.2f} ms/step (compile {compile_time:.1f}s)  loss {losses[0]:.4f} -> {losses[-1]:.4f}")
except Exception as e:
    print(f"compiled AMP: FAIL {type(e).__name__}: {str(e)[:300]}")
