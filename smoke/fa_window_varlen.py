import sys
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zluda_ops import flash_attn_func, flash_attn_varlen_func
try: torch.cuda.set_per_process_memory_fraction(0.85)
except Exception as e: print("vram", e, flush=True)
torch.manual_seed(0)
# sliding window causal
q=torch.randn(1,2,128,64,device="cuda",dtype=torch.float16,requires_grad=True)
k=torch.randn_like(q,requires_grad=True); v=torch.randn_like(q,requires_grad=True)
o=flash_attn_func(q,k,v,causal=True,window_size=(32,0))
qi=torch.arange(128,device="cuda")[:,None]; ki=torch.arange(128,device="cuda")[None,:]
allowed=(qi>=ki)&((qi-ki)<=32); mask=~allowed
with sdpa_kernel(SDPBackend.MATH): r=F.scaled_dot_product_attention(q,k,v,attn_mask=mask)
print("window fwd err",(o.float()-r.float()).abs().max().item())
o.backward(torch.randn_like(o)); print("window grad ok", q.grad.abs().max().item())
# varlen
q=torch.randn(192,2,64,device="cuda",dtype=torch.float16,requires_grad=True)
k=torch.randn_like(q,requires_grad=True); v=torch.randn_like(q,requires_grad=True)
cu=torch.tensor([0,64,192],device="cuda",dtype=torch.int32)
o=flash_attn_varlen_func(q,k,v,cu,cu,causal=True)
refs=[]
for a,b in ((0,64),(64,192)):
    qi=q[a:b].transpose(0,1).unsqueeze(0); ki=k[a:b].transpose(0,1).unsqueeze(0); vi=v[a:b].transpose(0,1).unsqueeze(0)
    with sdpa_kernel(SDPBackend.MATH): refs.append(F.scaled_dot_product_attention(qi,ki,vi,is_causal=True)[0].transpose(0,1))
r=torch.cat(refs)
print("varlen fwd err",(o.float()-r.float()).abs().max().item())
o.float().mean().backward(); print("varlen grad ok", q.grad.abs().max().item())
print("OK window/varlen")
