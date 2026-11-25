import torch
print("mha_forward", torch.nn.MultiheadAttention.forward.__module__, torch.nn.MultiheadAttention.forward.__name__, flush=True)
print("vram", [round(x/1024**3,2) for x in torch.cuda.mem_get_info()], flush=True)
m=torch.nn.MultiheadAttention(128,4,batch_first=True).cuda().train()
q=torch.randn(8,32,128,device="cuda"); k=torch.randn_like(q); v=torch.randn_like(q)
o=m(q,k,v,need_weights=False)
o.float().mean().backward()
print("mha_ok", o.shape, flush=True)
