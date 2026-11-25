import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import torch.nn as nn
try: torch.cuda.set_per_process_memory_fraction(0.85)
except Exception: pass
from zluda_ops import mha
mha.install()
torch.manual_seed(0)
m=nn.TransformerEncoderLayer(128,4,256,dropout=0.1,batch_first=True,norm_first=True).cuda().train()
x=torch.randn(8,32,128,device="cuda"); mask=torch.zeros(8,32,dtype=torch.bool,device="cuda")
def step(model):
    with torch.autocast("cuda",dtype=torch.float16):
        y=model(x,src_key_padding_mask=mask); loss=y.float().pow(2).mean()
    loss.backward(); return float(loss.detach().cpu())
print("eager",step(m),flush=True)
t0=time.perf_counter()
try:
    c=torch.compile(m,dynamic=True,fullgraph=False)
    v=step(c)
    print("compiled",v,"sec",time.perf_counter()-t0,flush=True)
except Exception as e:
    print("COMPILE_FAIL",type(e).__name__,str(e)[:500],flush=True)
