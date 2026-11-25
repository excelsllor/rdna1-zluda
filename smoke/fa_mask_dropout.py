import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zluda_ops import scaled_dot_product_attention as sdpa
try: torch.cuda.set_per_process_memory_fraction(0.85)
except Exception: pass
torch.manual_seed(0)
q=torch.randn(1,8,128,64,device="cuda",dtype=torch.float16,requires_grad=True)
k=torch.randn_like(q,requires_grad=True); v=torch.randn_like(q,requires_grad=True)
mask=torch.rand(128,128,device="cuda") > 0.2
o=sdpa(q,k,v,attn_mask=mask,dropout_p=0.0)
print("mask fwd",o.shape,flush=True)
o.float().mean().backward(); print("mask grad",q.grad.abs().max().item(),flush=True)
q.grad=k.grad=v.grad=None
o=sdpa(q,k,v,attn_mask=None,dropout_p=0.1); o.float().mean().backward()
print("dropout fwd+bwd ok",q.grad.abs().max().item(),flush=True)
print("OK mask/dropout",flush=True)
