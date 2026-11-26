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
## Final validation (2026-11-26, bench/phase2_kernels.py: 9/9 OK)

| kernel | result |
|---|---|
| GQA KV-reuse decode vs head-per-program | 91.9 vs 122.7 us -> **1.34x**, err 5.6e-5 |
| M=1 specialised decode | 84.7 vs 113.8 us -> **1.34x**, err 4.9e-5 (M>1 path now also exact: 6.9e-5) |
| fused residual+RMSNorm | err 1e-3; 38.4 vs 52.6 us in the bench (1.37x; the kernel itself is ~1 us, the rest is launch overhead) |
| fused QKV+RoPE+KV-append | q/k/v err 1.8e-3 / 9e-4 / 1e-3 |
| embedding gather+scale | exact |
| sampling (Gumbel-max) | greedy exact, top-50 hits 40/40, distribution matches softmax |
| int8 prefill GEMM | err 0.004; 475.6 us vs 242.0 us dequant-once+fp16 GEMM (correct, but the 8-bit tile kernel is slower on RDNA1: no int8 dot, keep as a VRAM-saving option only) |

**End-to-end (GraphDecoder, 1k context):** with the fused decoder-layer norm and
fused MLP wired in via `zluda_ops/model_patches.py`, replay drops from 11.02 to
10.36 ms/token (**90.8 -> 96.6 tok/s, +6.4%**), logits max err 0.034 (fp16).

**Roofline (bench/roofline.py, peak 448 GB/s):** copy 379 GB/s (85%), fill
425 GB/s (95%), GEMV 4864x896 154 GB/s (34%), 896x4864 177 GB/s (39%),
lm_head 151936x896 347 GB/s (77%); small decode kernels are dominated by the
~25 us launch/dispatch floor rather than bandwidth.

## Bugs found and fixed while validating
- `add_rms_norm` with `N: tl.constexpr` triggered a Triton/ZLUDA codegen bug
  (kernel wrote only 128 elements per row for N=896). Fixed by passing N as a
  runtime argument; all fused kernels now avoid shape-dependent constexpr N.
- The M>1 GQA "mismatch" was a bad reference in the bench (PyTorch SDPA bool
  mask semantics), not a kernel bug; the bench now uses an explicit torch-math
  reference and M>1 is exact.
- `tl.cumsum` is unreliable on this stack, so the sampler uses the Gumbel-max
  trick (no cumsum, no probability vector).

## Remaining
- int8 prefill: keep for VRAM-bound cases, tune tiles or drop in favour of
  dequant-once + fp16 GEMM on this GPU;
- sampling kernel is correct but single-program (658 us for a 151936 vocab):
  parallelise the block reduction before using it in hot loops;
- autotune tables per shape; wire fused QKV/GQA kernels into the graph-decoder
  layer path (currently fused norm + MLP).