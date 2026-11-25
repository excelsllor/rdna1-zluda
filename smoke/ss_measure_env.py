import os,time,torch
n=int(os.environ.get("N","4096")); its=int(os.environ.get("ITERS","5"))
a=torch.randn(n,n,device="cuda",dtype=torch.float32); b=torch.randn(n,n,device="cuda",dtype=torch.float32)
for _ in range(2): c=a@b
torch.cuda.synchronize(); t=time.perf_counter()
for _ in range(its): c=a@b
torch.cuda.synchronize(); ms=(time.perf_counter()-t)/its*1000
print(round(ms,4),round(2*n**3/(ms/1000)/1e12,3),flush=True)
