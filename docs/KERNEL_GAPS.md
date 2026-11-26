# Kernel gaps: what should exist and does not (2026-11-26)

Audit of kernels that standard GPU stacks provide (cuBLAS/rocBLAS, PyTorch,
flash-attn/xformers) but our ZLUDA/RDNA1 stack is missing or implements
sub-optimally, with the status and the plan per item.

## 1. Added in this round

### Fused int8 GEMV (`zluda_ops.quant.int8_linear`)
Before: int8 weights were supported only via "dequantize once + fp16 GEMV"
(no fused kernel). Now `_gemv_int8_kernel` dequantizes groupwise int8 inside the
matvec kernel (group 128, fp16 activations, M=1..16).

Measured (M=1, fp16): correctness vs dequantized reference 2e-4..4e-3 (int8
quantization noise). Performance vs dequant-once + tuned fp16 GEMV:

| shape | fused int8 | dequant-once + GEMV |
|---|---|---|
| 896 x 896 | 61.9 us | 45.6 us |
| 4864 x 896 | 123.1 us | 69.5 us |
| 896 x 4864 | 88.7 us | 74.5 us |

Verdict (same as int4): for compute at decode the dequant-once + tuned fp16
GEMV is faster; the fused int8 kernel is the memory-optimal option (1 byte per
weight, no fp16 copy). Keep both; pick per use case (VRAM-bound vs speed-bound).

### Fused gate/up + SwiGLU GEMV - merged (`zluda_ops.fused.swiglu_gemv`)
Computes `silu(x.Wg + bg) * (x.Wu + bu)` for M=1 decode in one kernel instead of
two GEMV calls plus a separate swiglu kernel.

Correctness (vs the 3-kernel reference): max abs error 0.0078 (fp16 rounding).
Performance (fp16, M=1):

| shape | fused | 3-kernel path | speedup |
|---|---|---|---|
| 4864 x 896 (gate/up) | 70.9 us | 98.3 us | 1.39x |
| 896 x 896 | 32.5 us | 70.5 us | 2.17x |

The first prototype returned NaN/large errors; the root cause was not a Triton
codegen bug but a missing K-bound mask (the last BK tile read past column K).
With the mask the kernel is correct; integrated into the Qwen2 MLP decode path
via `zluda_ops/model_patches.py` it gives a small end-to-end gain in
graph-replayed decode (~1%: 10.83 -> 10.74 ms/token at 1k context; the
kernel-level win is partly hidden by attention/GEMV/launch costs of the other
layers).

## 2. Missing kernels (not implemented yet, ranked by value)

| kernel | why it matters | status/plan |
|---|---|---|
| GQA KV-reuse flash-decoding | current flash-decoding reads K/V once per query head; with GQA (14 q / 2 kv) that is 7x redundant bandwidth on decode attention (23% of peak) | phase 2: program per (kv head, split), loop over query heads |
| fused QKV projection + RoPE + KV-cache append | 3-4 launches + intermediate writes per layer in decode | phase 2: one decode kernel per layer |
| fused residual + RMSNorm | every transformer layer does add + norm as two ops | phase 2: fold into one bandwidth-bound kernel |
| sampling kernels (top-k/top-p/typical) | generation loops use torch sort/topk + multinomial; a fused sampler removes syncs | phase 2 |
| embedding gather + scale | memory-bound gather currently torch index_select | phase 2 |
| int4/int8 fused prefill on RDNA1 tiles | covered by `quant.int4_matmul`; int8 fused prefill missing | phase 2 |
| conv2d Triton/Winograd | MIOpen fp32 Winograd already used via solver auto-selection; a Triton version is not expected to beat it | low priority |
| batched/strided small GEMM | rocBLAS covers it; only relevant for attention batching | low priority |

## 3. Library-level gaps found by the infra probe

- **cublasSgemv (1-D fp32 GEMV)** returns CUBLAS_STATUS_NOT_SUPPORTED
  (`torch.mv(x_fp32)` fails); 2-D fp32 GEMM works. fp16 GEMV works.
  This is a real BLAS L2 gap (used by some classic code paths).
- **fp64 GEMM** (cublasDgemm) unsupported (no f64 Tensile coverage for gfx1010).
- cuDNN batchnorm/RNN unmapped (fallbacks provided, see docs/INFRA_FIXES.md).

## 4. hipMalloc vs cudaMalloc on memory-bound kernels

Question: does allocating through HIP directly (`hipMalloc`) instead of the
CUDA runtime (`cudaMalloc`, mapped by ZLUDA) change memory-bound performance?

Measured on 256 MB buffers and a bandwidth-bound GEMV (lm_head 151936x896):

| operation | via torch (cudaMalloc path) | via HIP (hipMalloc) |
|---|---|---|
| allocation latency (256 MB) | 0.254 ms | 0.212 ms |
| fill (memset) | 0.64 ms = 421 GB/s | 0.63 ms = 426 GB/s |
| copy (D2D) | 1.40 ms = 384 GB/s | 1.40 ms = 383 GB/s |
| HIP GEMV with 272 MB weights | 0.76 ms = 358 GB/s | 0.75 ms = 358 GB/s |

Conclusion: under ZLUDA `cudaMalloc` **is** `hipMalloc` - the same device memory,
same pointers, identical kernel bandwidth (~84-95% of the 448 GB/s peak for
fill/copy/GEMV). Only the allocation call overhead differs slightly (torch's
caching allocator adds ~0.04 ms per 256 MB allocation); for steady-state decode
with reuse there is no benefit in switching. Use torch allocations everywhere;
direct hipMalloc only makes sense to bypass torch for one-off huge buffers.