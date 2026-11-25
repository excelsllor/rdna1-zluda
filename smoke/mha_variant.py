import os, torch, torch.nn as nn
variant = os.environ.get("VARIANT", "default")
try:
    torch.cuda.set_per_process_memory_fraction(0.85)
except Exception as e:
    print("vram_limit_err", e, flush=True)
if variant == "nocudnn":
    torch.backends.cudnn.enabled = False
elif variant == "flags":
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.manual_seed(0)
layer = nn.TransformerEncoderLayer(128, 4, 256, dropout=0.1, batch_first=True, norm_first=True).cuda().train()
x = torch.randn(8, 32, 128, device="cuda")
mask = torch.ones(8, 32, dtype=torch.bool, device="cuda")
opt = torch.optim.AdamW(layer.parameters(), lr=1e-4)
print("START", variant, flush=True)
for i in range(3):
    opt.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.float16):
        y = layer(x, src_key_padding_mask=mask)
        loss = y.float().pow(2).mean()
    loss.backward()
    opt.step()
    print("STEP", i, float(loss.detach().cpu()), flush=True)
print("OK", variant, flush=True)
