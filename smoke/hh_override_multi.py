import torch, time
torch._C._set_cublas_allow_fp16_accumulation(True)
for n in (1024,2048,4096):
 a=torch.randn(n,n,device="cuda",dtype=torch.float16); b=torch.randn(n,n,device="cuda",dtype=torch.float16)
 for _ in range(2): c=a@b
 torch.cuda.synchronize(); t=time.perf_counter()
 it=20 if n<=1024 else 10
 for _ in range(it): c=a@b
 torch.cuda.synchronize(); ms=(time.perf_counter()-t)/it*1000
 print(n,round(ms,4),round(2*n**3/(ms/1000)/1e12,3),flush=True)
