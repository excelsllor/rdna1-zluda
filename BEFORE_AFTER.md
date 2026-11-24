# Before / after

GPU: AMD Radeon RX 5700 XT (gfx1010:xnack-, 40 CU, 8 GB).
"Before" is the stock ROCm 10.1 user space with fallback rocBLAS kernels, or
the only available PyTorch path (math SDPA / rocBLAS GEMV). "After" is this
fork: ZLUDA gfx1010 fixes, tuned rocBLAS 7.1.1, Triton GEMV and Triton causal
flash attention. Full context in `RESULTS_RDNA1.md`; plans in `ROADMAP.md`.

## Headline

| Area | Metric | Before | After | Change |
|---|---|---|---|---|
| GEMM 4096 | fp32 | 28.3 ms / 4.85 TF | 16.8 ms / 8.18 TF | **1.69x** |
| GEMM 4096 | fp16 (fp32 accum) | 32.6 ms / 4.21 TF | 19.7 ms / 6.96 TF | **1.65x** |
| GEMM 4096 | bf16 native | 59.3 ms / 2.32 TF | 31.1 ms / 4.42 TF | **1.91x** |
| GEMM 4096 | bf16 via fp32 emulation | 59.3 ms / 2.32 TF | 17.35 ms / 7.92 TF | **3.41x** |
| GEMV 4096, M=1 | fp16 | 0.296 ms / 114 GB/s | 0.114 ms / 294 GB/s | **2.58x** |
| GEMV 4096, M=1 | fp32 | 0.433 ms / 155 GB/s | 0.216 ms / 311 GB/s | **2.01x** |
| GEMV 4096, M=4 | fp16 | 0.315 ms / 107 GB/s | 0.121 ms / 278 GB/s | **2.60x** |
| Causal SDPA 512 | fp16 fwd | 0.84 ms | 0.48 ms | **1.75x** |
| Causal SDPA 2048 | fp16 fwd | 7.54 ms | 3.11 ms | **2.42x** |
| Causal SDPA 512, GQA H=8/KV=2 | fp16 fwd | 0.85 ms | 0.54 ms | **1.57x** |

## GEMV / skinny GEMM (N=K=4096, fp16, Triton `bench/gemv.py`)

| M | Before: rocBLAS `F.linear` | After: Triton | Change | % of 448 GB/s |
|---|---|---|---|---|
| 1 | 114 GB/s | 294 GB/s | 2.58x | 66% |
| 4 | 107 GB/s | 278 GB/s | 2.60x | 62% |
| 8 | 109 GB/s | 242 GB/s | 2.22x | 54% |
| 16 | 108 GB/s | 155 GB/s | 1.44x | 35% |
| 32 | 76 GB/s | 78 GB/s | 1.03x | 17% |

`torch.mv`/`cublasSgemv` is unsupported for fp32, so GEMV had no usable cuBLAS
path before. M=1/4/8 reach 78–82% of the measured 377 GB/s copy ceiling.
M=16/32 are reduction/compute-bound; for fp32 M=32 rocBLAS (112 GB/s) remains
faster than the Triton kernel (73 GB/s). Remaining work: packed-half
reductions, split-K, dwordx4 unpacking.

## Other measured results

| Area | Before | After | Note |
|---|---|---|---|
| GEMM 4096 shape grid | 8.16/6.92/4.40 TF (fp32/fp16/bf16) | same tuned path | Tensile kernels already in rocBLAS 7.1.1 |
| Batched GEMM b=8, 1024³ | - | 7.52 TF fp32 / 7.05 TF fp16 | strided: 7.51 / 6.90 TF |
| Bias epilogue | separate add 0.028 ms | cublasLt fused 0.877 ms vs GEMM 0.827 ms | fused is correct but not faster |
| Residual / GELU | unfused | unfused | 0.907 / 0.883 ms fp32 |
| int8 | unsupported | unsupported | `CUBLAS_STATUS_INVALID_VALUE` |
| Correctness gate | none | 12/12 checks + JSON reports | roofline/regression suite |
| Clocks / power | unavailable | unavailable | ZLUDA NVML NotSupported; no rocm-smi on Windows |

## What changed

1. ZLUDA gfx1010 build fixes: strip `:xnack-` from the LLVM target, disable
   DIA/PDB, `dp4a` software fallback, relaxed PTX `ld`/`st` cache hints,
   `ldmatrix.m8n8.x2.trans.b16` IR shim, `isa_version` in the module cache key.
2. cuBLASLt→rocBLAS fallback with descriptor tracking, per-row bias and
   fp16-compute descriptors.
3. Runtime: unified ROCm 10.1 user space + tuned rocBLAS 7.1.1 gfx1010
   Tensile libraries (the stock fallback kernels are the "before" GEMM).
4. Stage 0: `bench/roofline.py`, `bench/regression.py`; 12/12 correctness
   pass; FP32 ALU 5.0 TF, FP16 ALU 18.8 TF, memory 373–378 GB/s, HBM latency
   172 ns; rebuilt the missing Triton `ptxas` stub.
5. Stage 1: `bench/gemv.py` (M=1..32), `bench/stage1.py` (shape grid,
   batched/strided GEMM, epilogues, bf16, int8), `zluda_ops/bf16.py`.
6. Stage 2.1: `zluda_ops/flash_attention.py` + `bench/flash_pack.py`:
   autograd causal FA, SDPA-compatible function, GQA; non-causal backward and
   sliding/varlen/softcap/dropout fall back or remain open.
7. Repository cleanup: true fork of `vosen/ZLUDA`, only `main`,
   `rdna1-gfx1010` and upstream `master`; HIP 6.4 stack/CI removed; ComfyUI
   model paths shared and GGUF enabled.

## Not yet done

- New Tensile kernels for fp32 8.8–9.2 TF and HH 14–16 TF.
- GEMV M=16/32 at the 70–80% bandwidth target; split-K/flash-decoding.
- Non-causal Triton FA backward; sliding window, varlen, softcap, dropout.
- Fused activation/residual epilogues; int8 (Stage 3).
- GPU clock/power telemetry on Windows.
