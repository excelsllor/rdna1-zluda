import sys, time, math
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zluda_ops import scaled_dot_product_attention, flash_attn_func, FlashAttention
from bench.common import bench, save_report

def math_sdpa(q,k,v,causal):
    with sdpa_kernel(SDPBackend.MATH):
        return F.scaled_dot_product_attention(q,k,v,is_causal=causal)

def run_case(s=512, hq=8, hkv=8, d=64, causal=True, gqa=False):
    torch.manual_seed(0)
    q=torch.randn(1,hq,s,d,device="cuda",dtype=torch.float16,requires_grad=True)
    k=torch.randn(1,hkv,s,d,device="cuda",dtype=torch.float16,requires_grad=True)
    v=torch.randn(1,hkv,s,d,device="cuda",dtype=torch.float16,requires_grad=True)
    do=torch.randn_like(q)
    out=scaled_dot_product_attention(q,k,v,is_causal=causal)
    torch.cuda.synchronize()
    ke=k.repeat_interleave(hq//hkv,dim=1); ve=v.repeat_interleave(hq//hkv,dim=1)
    ref=math_sdpa(q,ke,ve,causal)
    ferr=(out.float()-ref.float()).abs().max().item()
    out.backward(do, retain_graph=True)
    gq=q.grad.clone(); gk=k.grad.clone(); gv=v.grad.clone()
    q.grad=k.grad=v.grad=None
    ref.backward(do)
    gerr=max((gq-q.grad).abs().max().item(),(gk-k.grad).abs().max().item(),(gv-v.grad).abs().max().item())
    q.grad=k.grad=v.grad=None
    ms_f=bench(lambda: scaled_dot_product_attention(q,k,v,is_causal=causal), warmup=3, iters=10)
    def fb():
        o=scaled_dot_product_attention(q,k,v,is_causal=causal); o.backward(do); q.grad=k.grad=v.grad=None
    ms_fb=bench(fb,warmup=2,iters=5)
    ms_ref=bench(lambda: math_sdpa(q,ke,ve,causal),warmup=2,iters=5)
    print(f"s={s} hq={hq} hkv={hkv} causal={causal}: fwd {ms_f:7.2f} ms ref {ms_ref:7.2f} | fwd+bwd {ms_fb:7.2f} | err {ferr:.4f} grad {gerr:.4f}")
    return {"s":s,"hq":hq,"hkv":hkv,"causal":causal,"fwd_ms":ms_f,"ref_ms":ms_ref,"fwd_bwd_ms":ms_fb,"fwd_err":ferr,"grad_err":gerr}

rows=[]
rows.append(run_case(512,8,8,64,True,False))
rows.append(run_case(1024,8,8,64,True,False))
rows.append(run_case(512,8,8,64,False,False))
rows.append(run_case(512,8,2,64,True,True))
# fallback path
torch.manual_seed(0)
q=torch.randn(1,4,512,32,device="cuda",dtype=torch.float16)
k=torch.randn_like(q); v=torch.randn_like(q)
o=scaled_dot_product_attention(q,k,v,is_causal=True); r=math_sdpa(q,k,v,True)
print("fallback d=32 err", (o.float()-r.float()).abs().max().item())
m=FlashAttention(causal=True)
o=m(q,k,v); print("module ok", o.shape)
save_report("flash_pack", {"rows": rows})
