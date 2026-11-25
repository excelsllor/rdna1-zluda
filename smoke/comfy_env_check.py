import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0))
import torchvision
print("torchvision", torchvision.__version__)
import triton
print("triton", triton.__version__)
