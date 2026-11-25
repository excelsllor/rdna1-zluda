import torch
print([n for n in dir(torch._C) if 'fp16' in n.lower() or 'accum' in n.lower()])
for n in ("_get_cublas_allow_fp16_accumulation","_set_cublas_allow_fp16_accumulation"):
    print(n, hasattr(torch._C,n))
