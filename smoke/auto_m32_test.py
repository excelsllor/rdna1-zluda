import torch, torch.nn.functional as F
try: torch.cuda.set_per_process_memory_fraction(0.85)
except Exception: pass
for dtype in (torch.float16, torch.float32):
    w=torch.randn(4096,4096,device="cuda",dtype=dtype)*0.02
    b=torch.randn(4096,device="cuda",dtype=dtype)*0.02
    for m in (16,32):
        x=torch.randn(m,4096,device="cuda",dtype=dtype)*0.1
        y=F.linear(x,w,b)
        import zluda_ops.auto as auto
        ref=auto._orig_linear(x,w,b)
        print(dtype,m,"err",(y.float()-ref.float()).abs().max().item(),"patched",F.linear.__module__,flush=True)
print("OK auto32",flush=True)
