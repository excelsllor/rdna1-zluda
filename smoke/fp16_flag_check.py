import torch
print("fp16_accum", torch._C._get_cublas_allow_fp16_accumulation(), flush=True)
