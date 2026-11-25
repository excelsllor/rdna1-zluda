import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
print("torch", torch.__version__, "cudnn_sdp", getattr(torch.backends.cuda, "cudnn_sdp_enabled", lambda: "n/a")())
torch.manual_seed(0)
q=torch.randn(8,4,32,32,device="cuda",dtype=torch.float16,requires_grad=True)
k=torch.randn_like(q,requires_grad=True); v=torch.randn_like(q,requires_grad=True)
mask=torch.zeros(8,1,1,32,device="cuda",dtype=torch.bool)
def run(name):
    try:
        o=F.scaled_dot_product_attention(q,k,v,attn_mask=mask,dropout_p=0.1,is_causal=False)
        loss=o.float().mean(); loss.backward()
        print(name, "OK", o.shape, flush=True)
    except Exception as e:
        print(name, "FAIL", type(e).__name__, str(e)[:300], flush=True)
    q.grad=k.grad=v.grad=None
run("default")
try:
    torch.backends.cuda.enable_cudnn_sdp(False)
    print("cudnn disabled", flush=True)
except Exception as e:
    print("disable cudnn err", e, flush=True)
run("no_cudnn")
with sdpa_kernel(SDPBackend.MATH):
    run("math")
