import torch
print("mha", dir(torch.backends.mha))
for name in ("get_fastpath_enabled","set_fastpath_enabled"):
    print(name, hasattr(torch.backends.mha,name))
print("cudnn_sdp", getattr(torch.backends.cuda,"cudnn_sdp_enabled",lambda:"n/a")())
