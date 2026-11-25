import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    torch.cuda.set_per_process_memory_fraction(0.85)
except Exception as e:
    print("vram_limit_err", e, flush=True)


def math_mha_forward(self, query, key, value, key_padding_mask=None, need_weights=True,
                     attn_mask=None, average_attn_weights=True, is_causal=False):
    if not self.batch_first:
        query, key, value = query.transpose(0, 1), key.transpose(0, 1), value.transpose(0, 1)
    b, l, e = query.shape
    s = key.shape[1]
    h = self.num_heads
    d = e // h
    qkv = F.linear(query, self.in_proj_weight, self.in_proj_bias)
    qkv = qkv.reshape(b, l, 3, h, d).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    scores = torch.matmul(q, k.transpose(-2, -1)) * (d ** -0.5)
    if is_causal:
        cmask = torch.triu(torch.ones(l, s, dtype=torch.bool, device=query.device), diagonal=1)
        scores = scores.masked_fill(cmask, float("-inf"))
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            scores = scores.masked_fill(attn_mask, float("-inf"))
        else:
            scores = scores + attn_mask
    if key_padding_mask is not None:
        kpm = key_padding_mask.bool()
        scores = scores.masked_fill(kpm[:, None, None, :], float("-inf"))
    p = torch.softmax(scores, dim=-1)
    p = F.dropout(p, p=self.dropout, training=self.training)
    out = torch.matmul(p, v).transpose(1, 2).reshape(b, l, e)
    out = self.out_proj(out)
    if not self.batch_first:
        out = out.transpose(0, 1)
    return (out, None) if not need_weights else (out, p.mean(dim=1))


nn.MultiheadAttention.forward = math_mha_forward

torch.manual_seed(0)
layer = nn.TransformerEncoderLayer(128, 4, 256, dropout=0.1, batch_first=True, norm_first=True).cuda().train()
x = torch.randn(8, 32, 128, device="cuda")
mask = torch.ones(8, 32, dtype=torch.bool, device="cuda")
opt = torch.optim.AdamW(layer.parameters(), lr=1e-4)
print("START math_mha", flush=True)
for i in range(3):
    opt.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.float16):
        y = layer(x, src_key_padding_mask=mask)
        loss = y.float().pow(2).mean()
    loss.backward()
    opt.step()
    print("STEP", i, float(loss.detach().cpu()), flush=True)
print("OK math_mha", flush=True)
