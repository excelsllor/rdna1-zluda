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


def make_model():
    return Block().half().cuda()


def train_steps(model, optimizer, steps, x, y):
    losses = []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = F.mse_loss(model(x).float(), y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    torch.cuda.synchronize()
    return losses


x = torch.randn(B, S, D, device=dev, dtype=torch.float16)
y = torch.randn((), device=dev, dtype=torch.float32)

model = make_model()
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

# warmup
train_steps(model, opt, 3, x, y)
t0 = time.perf_counter()
losses = train_steps(model, opt, 20, x, y)
ms_eager = (time.perf_counter() - t0) / 20 * 1000
print(f"eager train: {ms_eager:.2f} ms/step  loss {losses[0]:.4f} -> {losses[-1]:.4f}")

model2 = make_model()
model2.load_state_dict(model.state_dict())
opt2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
compiled = torch.compile(model2)
t0 = time.perf_counter()
try:
    train_steps(compiled, opt2, 3, x, y)
    compile_time = time.perf_counter() - t0
    t0 = time.perf_counter()
    losses2 = train_steps(compiled, opt2, 20, x, y)
    ms_comp = (time.perf_counter() - t0) / 20 * 1000
    print(f"compiled train: {ms_comp:.2f} ms/step (compile+3 steps {compile_time:.1f}s)  loss {losses2[0]:.4f} -> {losses2[-1]:.4f}")
except Exception as e:
    print(f"compiled train: FAIL {type(e).__name__}: {str(e)[:300]}")
