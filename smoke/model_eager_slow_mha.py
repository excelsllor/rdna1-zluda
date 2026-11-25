import os, sys, time
import torch
root = os.path.dirname(os.environ["MODEL_DIR"])
os.chdir(root); sys.path.insert(0, root)
try: torch.cuda.set_per_process_memory_fraction(0.85)
except Exception as e: print("vram_limit_err", e, flush=True)
_orig = torch.nn.MultiheadAttention.forward
def slow_mha(self, *args, **kwargs):
    need = kwargs.get("need_weights", True)
    kwargs = dict(kwargs); kwargs["need_weights"] = True
    torch.backends.mha.set_fastpath_enabled(False)
    out, weights = _orig(self, *args, **kwargs)
    return (out, weights) if need else out
torch.nn.MultiheadAttention.forward = slow_mha
from avito.transformer_probe import FeatureSetRanker
B,N,T=8,32,64
def batch(seed=7):
    g=torch.Generator().manual_seed(seed)
    d={"cand_feat":torch.randn(B,N,72,generator=g),"cand_emb":torch.randn(B,N,128,generator=g),
       "cand_cat":torch.randint(1,54,(B,N),generator=g),"cand_f":torch.randn(B,N,5,generator=g),
       "cand_mask":torch.ones(B,N,dtype=torch.bool),"hist_emb":torch.randn(B,T,128,generator=g),
       "hist_cat":torch.randint(1,54,(B,T),generator=g),"hist_type":torch.randint(1,11,(B,T),generator=g),
       "hist_src":torch.randint(1,31,(B,T),generator=g),"hist_mask":torch.ones(B,T,dtype=torch.bool),
       "user_emb":torch.randn(B,128,generator=g),"target":torch.randint(0,2,(B,N),generator=g).float()}
    return {k:v.cuda() for k,v in d.items()}
x=batch(); m=FeatureSetRanker(72,d=128,heads=4,layers=8,max_hist=64,dropout=0.1,use_bdia=False,use_cfc=True,cfc_mode="history",feature_encoder="grouped",attention_impl="torch",cfc_layers=(2,7)).cuda().train()
opt=torch.optim.AdamW(m.parameters(),lr=2e-4,weight_decay=0.01)
print("START eager_slow_mha",flush=True)
for i in range(2):
    opt.zero_grad(set_to_none=True)
    t0=time.perf_counter()
    with torch.autocast("cuda",dtype=torch.float16):
        out=m(**{k:v for k,v in x.items() if k!="target"}); loss=out.float().pow(2).mean()
    loss.backward(); opt.step(); torch.cuda.synchronize()
    print("STEP",i,"loss",float(loss.detach().cpu()),"ms",(time.perf_counter()-t0)*1000,flush=True)
print("OK eager_slow_mha",flush=True)
