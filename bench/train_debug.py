import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
dev = "cuda"


def report(name, t):
    if isinstance(t, torch.Tensor):
        d = t.detach().float()
        print(f"{name}: shape {tuple(d.shape)} nan={torch.isnan(d).any().item()} inf={torch.isinf(d).any().item()} max={d.abs().max().item():.4g}")


# simple MLP training
for dtype in (torch.float32, torch.float16):
    torch.manual_seed(0)
    mlp = nn.Sequential(nn.Linear(256, 512), nn.GELU(), nn.Linear(512, 1)).to(dev, dtype)
    opt = torch.optim.AdamW(mlp.parameters(), lr=1e-3)
    x = torch.randn(64, 256, device=dev, dtype=dtype)
    y = torch.randn(64, 1, device=dev, dtype=dtype)
    out = mlp(x)
    report(f"mlp[{dtype}] out", out)
    loss = F.mse_loss(out, y)
    print(f"mlp[{dtype}] loss {loss.item()}")
    loss.backward()
    g = [p.grad for p in mlp.parameters() if p.grad is not None]
    bad = sum(torch.isnan(gg).any().item() for gg in g)
    print(f"mlp[{dtype}] grads with nan: {bad}/{len(g)}")
    opt.step()
    out2 = mlp(x)
    print(f"mlp[{dtype}] after step nan={torch.isnan(out2).any().item()}")

# attention block backward
B, S, H, HD = 8, 128, 8, 64
D = H * HD
for dtype in (torch.float32, torch.float16):
    torch.manual_seed(0)
    qkv = nn.Linear(D, 3 * D).to(dev, dtype)
    proj = nn.Linear(D, D).to(dev, dtype)
    x = torch.randn(B, S, D, device=dev, dtype=dtype)
    q, k, v = qkv(x).view(B, S, 3, H, HD).permute(2, 0, 3, 1, 4)
    att = (q @ k.transpose(-2, -1)) * (HD**-0.5)
    mask = torch.triu(torch.ones(S, S, device=dev, dtype=torch.bool), diagonal=1)
    att = att.masked_fill(mask, float("-inf"))
    report(f"att[{dtype}] before softmax", att)
    att = torch.softmax(att, dim=-1)
    report(f"att[{dtype}] softmax", att)
    o = (att @ v).transpose(1, 2).reshape(B, S, D)
    report(f"att[{dtype}] out", o)
    loss = proj(o).float().pow(2).mean()
    print(f"att[{dtype}] loss {loss.item()}")
    loss.backward()
    bad = sum(torch.isnan(p.grad).any().item() for p in proj.parameters() if p.grad is not None)
    print(f"att[{dtype}] proj grads nan: {bad}")
    badq = sum(torch.isnan(qkv.weight.grad).any().item() for _ in [0] if qkv.weight.grad is not None)
    print(f"att[{dtype}] qkv grad nan: {badq}")
