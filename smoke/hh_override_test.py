import torch, time
torch._C._set_cublas_allow_fp16_accumulation(True)
n=4096
a=torch.randn(n,n,device="cuda",dtype=torch.float16); b=torch.randn(n,n,device="cuda",dtype=torch.float16)
for _ in range(2): c=a@b
torch.cuda.synchronize(); t=time.perf_counter()
for _ in range(5): c=a@b
torch.cuda.synchronize(); ms=(time.perf_counter()-t)/5*1000
print("HH",ms,2*n**3/(ms/1000)/1e12,flush=True)
ref=(a.cpu().double()@b.cpu().double()).float()
print("err",(c.cpu().double()-ref.double()).abs().max().item(),flush=True)
