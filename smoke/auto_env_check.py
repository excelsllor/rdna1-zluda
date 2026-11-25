import torch, torch.nn.functional as F
print("linear", getattr(F.linear, "__module__", ""), getattr(F.linear, "__name__", ""))
x=torch.randn(1,4096,device="cuda",dtype=torch.float16)
w=torch.randn(4096,4096,device="cuda",dtype=torch.float16)
print("out", F.linear(x,w).shape)
