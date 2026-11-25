import torch
names = ("utilization", "temperature", "clock_rate", "power_draw", "memory_usage")
print("available:", [n for n in names if hasattr(torch.cuda, n)])
for n in names:
    if hasattr(torch.cuda, n):
        try:
            print(n, getattr(torch.cuda, n)(0))
        except Exception as e:
            print(n, "ERR", type(e).__name__, str(e)[:120])
