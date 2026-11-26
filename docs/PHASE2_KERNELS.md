# Phase 2 kernels: custom Triton fast paths

Status: **implemented, GPU validation pending** (the GPU/driver is wedged after
the compatibility-matrix run and needs a reboot; run
`venv\Scripts\python.exe bench\phase2_kernels.py` under `run-zluda.cmd` right
after the reboot).

## Implemented

### 1. GQA KV-reuse flash-decoding (`zluda_ops.gqa_decode.gqa_decode_attn`)
Problem: `flash_decode_attn` maps one program per (batch, query head); with GQA
(14 q / 2 kv heads) each of the 7 query heads of a group re-reads the same K/V
from DRAM, i.e. up to 7x redundant KV traffic on long contexts (decode attention
was at ~23% of peak bandwidth).

Design: one program per **(batch, kv head) x KV-split**, looping over the query
heads of the group. The K/V chunk of each split is loaded once per head
iteration; the following iterations hit it in the 4 MB L2 (chunk bytes =
N/SPLITS * D * 2, tiny compared to L2), so DRAM traffic drops to ~1x while the
SPLITS * B * HKV programs still fill the CUs. Partials/merge reuse the existing
`_decode_merge` layout, so the kernel is a drop-in alternative to
`flash_decode_attn` (returns None for unsupported shapes).

Expected: lower DRAM traffic on GQA decode; measured speedup/effective
bandwidth is reported by `bench/phase2_kernels.py`.

### 2. Fused residual + RMSNorm (`zluda_ops.fused.add_rms_norm`)
Computes `res = x + delta` (returned for the next residual) and
`y = rms_norm(res) * weight` in a single bandwidth-bound kernel, removing one
kernel launch and one extra read/write of the residual stream per layer.

## Next Phase-2 kernels (planned)
- fused QKV projection + RoPE + KV-cache append (decode, one kernel/layer);
- fused MLP projections + activation (gate/up fused already; add down fusion);
- sampling kernels (temperature/top-k/top-p fused);
- embedding gather + scale;
- int8/int4 prefill tiles (fused GEMM with on-the-fly dequant);
- autotune tables (per-shape configs) and roofline harness for all decode
  kernels.
## Validated results (RX 5700 XT, fp16, bench/phase2_kernels.py)

| kernel | check | result |
|---|---|---|
| GQA KV-reuse decode (M=1 general) | correctness vs math SDPA | 3e-5 |
| GQA KV-reuse decode | speed vs head-per-program `flash_decode_attn` | 87.5 vs 115.9 us -> **1.32x** |
| M=1 specialised decode (no BLOCK_M=16 padding) | correctness / speed | err 2e-6, 89.6 vs 113.1 us -> **1.26x** |
| fused residual + RMSNorm | correctness / speed | err 1e-3, **1.0 vs 46.2 us -> 43x** |
| fused QKV + RoPE + KV append | correctness (q/k/v) | 1.8e-3 / 9e-4 / 1e-3 |
| embedding gather + scale | correctness / speed | err 0.0, 21.3 us |

Known issue: the general (M>1, e.g. speculative decode) path disagrees with the
math reference (err 3.26) - the M=1 paths (the dominant decode case) are exact
and 1.3x faster. M>1 needs a dedicated follow-up (mask/partial layout audit;
debug script hung once, so it is tracked as an open item, not wired into the
model path).

## Still to implement (Phase 2 remainder)
- sampling kernels (temperature/top-k/top-p fused);
- int8/int4 prefill tiles with on-the-fly dequant;
- autotune tables per shape and a roofline harness for all decode kernels;
- wire the validated kernels (add_rms_norm, GQA decode, fused QKV) into
  GraphDecoder / model_patches and measure the end-to-end decode gain.