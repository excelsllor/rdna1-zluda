import torch, triton, triton.language as tl, traceback
@triton.jit
def k(w_ptr, x_ptr, y_ptr, N, K: tl.constexpr, M: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(0); rows = pid * BN + tl.arange(0, BN)
    accs = [tl.zeros((BN,), tl.float32) for _ in range(M)]
    for k0 in range(0, K, BK):
        offs = k0 + tl.arange(0, BK)
        w = tl.load(w_ptr + rows[:, None] * K + offs[None, :])
        for mi in range(M):
            xv = tl.load(x_ptr + mi * K + offs)
            accs[mi] += tl.sum(w * xv[None, :], axis=1)
    for mi in range(M):
        tl.store(y_ptr + mi * N + rows, accs[mi])
w=torch.randn(64,64,device="cuda",dtype=torch.float16); x=torch.randn(4,64,device="cuda",dtype=torch.float16); y=torch.empty(4,64,device="cuda",dtype=torch.float16)
try:
    k[(32,)](w,x,y,64,K=64,M=4,BN=2,BK=64,num_warps=4); torch.cuda.synchronize(); print("ok")
except Exception as e:
    print(type(e).__name__, str(e)[:800]); traceback.print_exc(limit=3)
