import torch
import torch.nn as nn
try:
    torch.cuda.set_per_process_memory_fraction(0.85)
except Exception as e:
    print("vram_limit_err", e, flush=True)

_orig = nn.MultiheadAttention.forward
def slow_mha(self, *args, **kwargs):
    need_weights = kwargs.get("need_weights", True)
    kwargs = dict(kwargs)
    kwargs["need_weights"] = True
    torch.backends.mha.set_fastpath_enabled(False)
    out, weights = _orig(self, *args, **kwargs)
    return (out, weights) if need_weights else out
nn.MultiheadAttention.forward = slow_mha

torch.manual_seed(0)
layer = nn.TransformerEncoderLayer(128, 4, 256, dropout=0.1, batch_first=True, norm_first=True).cuda().train()
x = torch.randn(8, 32, 128, device="cuda")
mask = torch.zeros(8, 32, dtype=torch.bool, device="cuda")
opt = torch.optim.AdamW(layer.parameters(), lr=1e-4)
print("START slow_mha", flush=True)
for i in range(3):
    opt.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.float16):
        y = layer(x, src_key_padding_mask=mask)
        loss = y.float().pow(2).mean()
    loss.backward()
    opt.step()
    print("STEP", i, float(loss.detach().cpu()), flush=True)
print("OK slow_mha", flush=True)
