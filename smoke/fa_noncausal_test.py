import sys, time
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zluda_ops import scaled_dot_product_attention as sdpa
try: torch.cuda.set_per_process_memory_fraction(0.85)
except Exception as e: print("vram", e, flush=True)
for s in (512,1024):
    torch.manual_seed(0)
    q=torch.randn(1,8,s,64,device="cuda",dtype=torch.float16,requires_grad=True)
    k=torch.randn_like(q,requires_grad=True); v=torch.randn_like(q,requires_grad=True)
    do=torch.randn_like(q)
    o=sdpa(q,k,v,is_causal=False); torch.cuda.synchronize()
    with sdpa_kernel(SDPBackend.MATH): r=F.scaled_dot_product_attention(q,k,v,is_causal=False)
    ferr=(o.float()-r.float()).abs().max().item()
    o.backward(do,retain_graph=True); gq=q.grad.clone(); gk=k.grad.clone(); gv=v.grad.clone(); q.grad=k.grad=v.grad=None
    with sdpa_kernel(SDPBackend.MATH): r2=F.scaled_dot_product_attention(q,k,v,is_causal=False); r2.backward(do)
    gerr=max((gq-q.grad).abs().max().item(),(gk-k.grad).abs().max().item(),(gv-v.grad).abs().max().item())
    print(f"s={s} fwd_err={ferr:.5f} grad_err={gerr:.5f}",flush=True)
print("OK noncausal",flush=True)
