import torch, time
torch._C._set_cublas_allow_fp16_accumulation(True)
n=4096
def bench(name,a,b):
 c=torch.empty(n,n,device="cuda",dtype=torch.float16)
 for _ in range(3): c=a@b
 torch.cuda.synchronize(); t=time.perf_counter()
 for _ in range(8): c=a@b
 torch.cuda.synchronize(); ms=(time.perf_counter()-t)/8*1000
 print(name,ms,2*n**3/(ms/1000)/1e12,flush=True)
r1=torch.randn(n,n,device="cuda",dtype=torch.float16); r2=torch.randn(n,n,device="cuda",dtype=torch.float16)
z=torch.zeros(n,n,device="cuda",dtype=torch.float16); o=torch.ones(n,n,device="cuda",dtype=torch.float16); e1=torch.empty(n,n,device="cuda",dtype=torch.float16); e2=torch.empty(n,n,device="cuda",dtype=torch.float16)
bench("rand+rand",r1,r2); bench("rand+zero",r1,z); bench("zero+zero",z,z); bench("ones+ones",o,o); bench("empty+empty",e1,e2)
