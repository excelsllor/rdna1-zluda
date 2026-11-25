import sys, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zluda_ops import scaled_dot_product_attention as sdpa, flash_attn_func, FlashAttention
from bench.common import bench
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
torch.manual_seed(0)
q=torch.randn(1,8,2048,64,device="cuda",dtype=torch.float16,requires_grad=True)
k=torch.randn_like(q,requires_grad=True); v=torch.randn_like(q,requires_grad=True)
do=torch.randn_like(q)
o=sdpa(q,k,v,is_causal=True)
with sdpa_kernel(SDPBackend.MATH):
    r=F.scaled_dot_product_attention(q,k,v,is_causal=True)
print("fwd err",(o.float()-r.float()).abs().max().item())
o.backward(do); q.grad=None; k.grad=None; v.grad=None
with sdpa_kernel(SDPBackend.MATH):
    r=F.scaled_dot_product_attention(q,k,v,is_causal=True); r.backward(do)
print("grads done", q.grad.abs().max().item(), k.grad.abs().max().item())
print("fwd 2048 ms", bench(lambda: sdpa(q,k,v,is_causal=True), warmup=2, iters=5))
o2=flash_attn_func(q,k,v,causal=True); print("flash_attn_func", o2.shape)
m=FlashAttention(causal=True); print("module", m(q,k,v).shape)
