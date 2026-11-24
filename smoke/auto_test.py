import sys, time
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import zluda_ops.auto as auto

def bench(fn, warmup=5, iters=20):
    for _ in range(warmup): fn()
    torch.cuda.synchronize(); t=time.perf_counter()
    for _ in range(iters): fn()
    torch.cuda.synchronize(); return (time.perf_counter()-t)/iters*1000

def math_sdpa(q,k,v,causal):
    with sdpa_kernel(SDPBackend.MATH):
        return auto._orig_sdpa(q,k,v,is_causal=causal)

print("patched", auto.installed(), F.linear.__name__, F.scaled_dot_product_attention.__name__)
torch.manual_seed(0)
for dtype in (torch.float16, torch.float32):
    for kdim in (4096, 3200):
        w=torch.randn(4096,kdim,device="cuda",dtype=dtype)*0.02
        b=torch.randn(4096,device="cuda",dtype=dtype)*0.02
        for m in (1,4,8):
            x=torch.randn(m,kdim,device="cuda",dtype=dtype)*0.1
            y=F.linear(x,w,b); ref=auto._orig_linear(x,w,b)
            err=(y.float()-ref.float()).abs().max().item()
            ms=bench(lambda: F.linear(x,w,b)); ms_ref=bench(lambda: auto._orig_linear(x,w,b))
            print(f"{str(dtype).split('.')[-1]:8s} k={kdim} m={m}: {ms:.4f} vs {ms_ref:.4f} ms err={err:.5f}")
for n in (1024,4096):
    x=torch.randn(64,n,device="cuda",dtype=torch.bfloat16)
    w=torch.randn(n,n,device="cuda",dtype=torch.bfloat16)*0.02
    b=torch.randn(n,device="cuda",dtype=torch.bfloat16)*0.02
    y=F.linear(x,w,b); ref=auto._orig_linear(x,w,b)
    err=(y.float()-ref.float()).abs().max().item()
    ms=bench(lambda: F.linear(x,w,b),3,10); ms_ref=bench(lambda: auto._orig_linear(x,w,b),3,10)
    print(f"bf16 n={n}: {ms:.3f} vs {ms_ref:.3f} ms err={err:.4f}")
q=torch.randn(1,8,512,64,device="cuda",dtype=torch.float16); k=torch.randn_like(q); v=torch.randn_like(q)
o=F.scaled_dot_product_attention(q,k,v,is_causal=True); r=math_sdpa(q,k,v,True)
print("sdpa causal err", (o.float()-r.float()).abs().max().item(), "ms", bench(lambda: F.scaled_dot_product_attention(q,k,v,is_causal=True),3,10), "vs", bench(lambda: math_sdpa(q,k,v,True),3,10))
o=F.scaled_dot_product_attention(q,k,v,is_causal=False); r=math_sdpa(q,k,v,False)
print("sdpa noncausal err", (o.float()-r.float()).abs().max().item())
with torch.no_grad():
    y=F.linear(torch.randn(1,4096,device="cuda",dtype=torch.float16), torch.randn(4096,4096,device="cuda",dtype=torch.float16))
print("no_grad ok", y.shape)
