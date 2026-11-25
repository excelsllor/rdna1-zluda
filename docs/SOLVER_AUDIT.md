# Solver / kernel audit (RDNA1, 2026-11-25)

Goal: go through every computation in the stack, check which solver/kernel is
used, and replace non-optimal ones. GPU clocks are unfixed (+-15-30% spread).

## 1. GEMV (decode matrix-vector products)

Before: one fixed config `(BK=4096, warps=4)` for all small-M shapes; rocBLAS was
used in decode until the auto patch started matching decode layouts.

Sweep (M=1, fp16 weights/activations, gfx1010):

| shape (N x K) | role | rocBLAS | Triton before | Triton tuned (BK,warps) | speedup vs rocBLAS |
|---|---|---|---|---|---|
| 896 x 896 | qkv / o_proj | 59.0 us | ~35 us | **28.5 us (512, 2)** | 2.07x |
| 4864 x 896 | gate/up | 80.7 us | ~35 us | **40.1 us (1024, 2)** | 2.01x |
| 896 x 4864 | down_proj | 284.1 us | ~36 us | **53.7 us (512, 4)** | 5.29x |
| 151936 x 896 | lm_head | 9739.8 us | ~800 us | **798.0 us (1024, 2)** | 12.20x |

Action: `zluda_ops/gemv.py` now has a shape-aware `TUNED_M1` table
(`_config(m, n, k)`), falling back to `CONFIGS[m]`. This is what makes decode
graph-fast (lm_head used to be 9.7 ms of a 34 ms token before the GEMV path).

## 2. Flash-attention prefill (Triton FA)

Current: BLOCK_M=64, BLOCK_N=64, num_warps=4, num_stages=1 (causal).
Sweep over 10 configurations at seq 512/1024/2048/4096/8192 (14 heads, D=64):

| seq | current | best config | gain |
|---|---|---|---|
| 512 | 0.60 ms | (64,64,4,2) 0.57 ms | 1.05x |
| 1024 | 1.47 ms | (64,64,4,1) 1.47 ms | 1.00x |
| 2048 | 4.71 ms | (64,64,8,1) 4.69 ms | 1.00x |
| 4096 | 16.75 ms | (128,64,4,2) 16.59 ms | 1.01x |
| 8192 | 63.00 ms | (128,64,4,1) 62.96 ms | 1.00x |

Verdict: **current FA prefill config is optimal** (all differences within
noise). A `_CONFIG_OVERRIDE` hook was added for future autotuning.

## 3. Flash-decoding (decode attention, split-KV)

Current heuristic `_choose_splits(n_ctx, batch_heads)` picks the largest
SPLITS<=16 dividing N/128 and filling >=128 programs with BLOCK_N=128.

Sweep over SPLITS 1..32 at KV 512..8192 (GQA 14 q / 2 kv heads):

| KV len | math SDPA | heuristic SPLITS | Triton flash-decode | speedup vs math |
|---|---|---|---|---|
| 512 | 255.2 us | 4 | 82.1 us | 3.11x |
| 1024 | 405.4 us | 8 | 106.5 us | 3.81x |
| 2048 | 424.9 us | 16 | 140.2 us | 3.03x |
| 4096 | 530.4 us | 16 | 208.8 us | 2.54x |
| 8192 | 1897.3 us | 16 | 280.6 us | 6.76x |

Verdict: heuristic SPLITS are already optimal at every length (gain 1.00x over
the best swept value). **But the kernel did not support GQA**: the dispatcher
required equal query/KV head counts, so Qwen-style GQA decode fell back to math
SDPA. Added native GQA indexing (`off_hkv = off_h // GROUP`) to `_decode_split`
and relaxed the auto dispatch to `q_heads % kv_heads == 0`; correctness vs an
explicit repeat-interleave reference is 3e-5 at KV 512/2048. End-to-end
graph-replay decode with correct GQA flash-decoding: 89.7 tok/s @1k, 72.3 tok/s
@2k.

## 4. Other computations checked

| computation | current solver | verdict |
|---|---|---|
| GEMM (prefill matmuls) | rocBLAS 7.1.1 + tuned gfx1010 Tensile overrides (ROCBLAS_TENSILE_GEMM_OVERRIDE_PATH) | near peak (8.18 TF fp32 of 9.75 TF paper peak); native ROCm torch without overrides is 4.6x slower on fp32 - keep |
| RMSNorm / SwiGLU / GeGLU / RoPE / softmax / AdamW | zluda_ops.fused Triton kernels | 1.5-10.9x faster than torch eager (previous benchmarks), keep |
| fp16 accumulation | cuBLAS HPA path | 1.5-1.7x prefill; keep opt-in |
| bf16 native | ZLUDA bf16 | slower than fp32/emul; emul (fp32 compute) is the optimal bf16 route |
| bf16 emul | bf16 weights -> fp32 compute | 5488 tok/s prefill (vs 3450 native bf16) - optimal available |
| int4 GEMV/GEMM | zluda_ops.quant tuned for 4096 shapes | decode shapes need a dedicated sweep (open) |
| int8 | dequant-once cache (no fused kernel) | fused int8 GEMV is missing (open; memory-optimal but not compute-optimal) |
| 8-bit AdamW | Triton blockwise int8 | 4.6x faster than torch AdamW, keep |
| conv2d | MIOpen fp16 solver vs fp32 Winograd workaround | audit in progress (see next section) |
| FFT / sparse / RNG | cuFFT->rocFFT, cuSPARSE->rocSPARSE, cuRAND->? mapping | used only by non-LLM workloads; not yet audited |

## 5. Open items

- Conv2d solver selection (MIOpen fp16 GemmFwdRest vs fp32 Winograd) - implement
  per-shape solver cache/dispatch.
- int4 decoder-shaped GEMV sweep and fused int8 GEMV.
- GEMM epilogue/hipBLASLt alternatives are not exposed by ZLUDA; rocBLAS path
  stays the optimum available.
- CUDA graphs for decode are now the delivery mechanism for the GEMV path
  (see docs/CUDA_GRAPHS_ZLUDA.md).