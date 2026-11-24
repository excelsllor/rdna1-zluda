# ZLUDA on Radeon RX 5700 XT (gfx1010 / RDNA1) — build and benchmark results

Before/after summary: [`BEFORE_AFTER.md`](BEFORE_AFTER.md)

Date: 2026-11-24
GPU: AMD Radeon RX 5700 XT (Navi 10, gfx1010:xnack-, 40 CU / 2560 SP, 8 GB)
Theoretical peaks: **FP32 9.754 TFLOPS**, **FP16 19.51 TFLOPS** (packed 2:1).
Matrix units (tensor cores / MFMA): **none** on RDNA1.

## Software stacks

- **ZLUDA**: upstream `v7-preview.21` (commit `eb89ae3`) + the patches in this tree.
- **Runtime (supported)**:
  ROCm 10.1 user space from TheRock gfx1010 packages (HIP 7.16, MIOpen, rocFFT,
  rocSPARSE) **plus** rocBLAS 7.1.1 with tuned gfx1010 Tensile libraries
  (likelovewant ROCmLibs) and its matching `rocblas.dll`, replacing stock rocBLAS.
- **PyTorch**: `2.7.1+cu118` — the CUDA wheel branch that ships `compute_37` PTX,
  which ZLUDA consumes. `cu128` wheels are SASS-only and cannot be used.
- **Triton**: `triton-windows 3.3.1`, with `TRITON_PTXAS_PATH` pointing at ZLUDA's
  `ptxas.exe` so Triton produces PTX (not NVIDIA SASS).
- **Native ROCm baseline**: `torch 2.14.0+rocm10.1` (gfx1010 device packages).

## Local ZLUDA changes

1. `llvm_zluda`: strip `:xnack-` target features from `gcn_arch` (ISA parse +
   LLVM CPU name); `LLVM_ENABLE_DIA_SDK=OFF` (Build Tools has no ATL).
2. `ptx`: propagate `isa_version` into the emitter; software fallback for `dp4a`
   on targets without `v_dot4` (gfx1010); parser accepts and ignores `ld`/`st`
   cache policy / eviction priority / cache hint / prefetch size qualifiers
   (required by Triton and TorchInductor).
3. `llvm_zluda`: IR shim for `__zluda_ptx_impl_ldmatrix_m8n8_x2_trans_b16`,
   which is missing from the shipped `zluda_ptx_impl.bc` and is required by
   Triton attention backward (`tl.trans` operands).
4. `zluda_blaslt`: real cuBLASLt→rocBLAS fallback:
   descriptor tracking (layout/matmul), synthetic algorithm when hipBLASLt has
   no kernels for the GPU, `rocblas_gemm_ex` mapping (types, transposes,
   leading dimensions), per-row bias epilogue via a rank-1 update with a cached
   device ones buffer, and fp16 (compute 16F) descriptors forced to the
   fallback path.
5. Diagnostics for module compile errors; `isa_version` included in the cache key.

Runtime configuration (`run-zluda.cmd`): `ZLUDA_CC=8.6` (valid SM arch for
Triton; the default 8.5 does not exist), `TRITON_PTXAS_PATH`, isolated
`TRITON_CACHE_DIR`/`TORCHINDUCTOR_CACHE_DIR`, `HIP_PATH`.

## GEMM results (4096x4096)

| path | fp32 | fp16 (fp32 accum) | bf16 |
|---|---|---|---|
| ROCm 10.1 stock rocBLAS (fallback Tensile) | 28.3 ms / 4.85 TF | 32.6 ms / 4.21 TF | 59.3 ms / 2.32 TF |
| **ROCm 10.1 + tuned rocBLAS 7.1.1** | **16.8–16.9 ms / 8.13–8.18 TF** | **19.7–19.8 ms / 6.93–6.96 TF** | **31.1 ms / 4.42 TF** |
| native ROCm 10.1 torch | 32.8 ms / 4.19 TF | 42.2 ms / 3.25 TF | 57.7 ms / 2.38 TF |

- fp32 reaches **~84% of theoretical peak** — normal for a well-tuned GEMM.
- fp16 with fp32 accumulation is bounded by FP32 rate: the tuned HPA kernels use
  `v_fma_mix` (verified by disassembling the Tensile code objects), so ~7 TF is
  close to the achievable ceiling for that math mode.
- The 19.5 TFLOPS FP16 peak requires packed half2 FMA (`v_pk_fma_f16`) **with
  fp16 accumulation**, which exists in the `HH` Tensile kernels.

### Optional fp16 accumulation (inference; reduced precision)

Enabled in PyTorch with `torch._C._set_cublas_allow_fp16_accumulation(True)`
(maps to `CUBLAS_COMPUTE_16F`, selects the packed `HH` kernels):

| n | time | TFLOPS | max abs err vs fp64 |
|---|---|---|---|
| 1024 | 0.17 ms | 12.71 | 2.38 |
| 2048 | 1.66 ms | 10.37 | 3.53 |
| 4096 | 11.19 ms | 12.28 | 7.92 |

`linear 256x1024x1024`: 0.16 ms (err 0.028) vs 0.20 ms with fp32 accumulation.
Not suitable for training without validation.

## Attention (B=1, H=8, D=64, fp16, causal)

| s | Triton flash fwd | math SDPA fwd | Triton flash fwd+bwd | math SDPA fwd+bwd |
|---|---|---|---|---|
| 512 | 0.48 ms | 0.65 ms | 1.42 ms | 1.22 ms |
| 1024 | 1.08 ms | 1.79 ms | 3.70 ms | 3.73 ms |
| 2048 | 3.09 ms | 7.66 ms | 10.92 ms | 13.93 ms |

- Forward uses on-line softmax with `tl.dot` (ZLUDA fallback MMA on RDNA1); no
  `ldmatrix.trans` variants besides x2/x4 are required after the IR shim.
- Backward gradients match math SDPA within 0.002–0.004 (max abs).
- **FlexAttention** (`torch.compile`, s=512 causal block mask): **0.50 ms**,
  max err 0.002 vs math SDPA.
- `torch.compile` of a small transformer block: forward 8.32 → 7.84 ms.
- AMP training (small transformer block, AdamW): **65.3 ms/step on ZLUDA** vs
  **131.8 ms/step on native ROCm 10.1** (fp16 AMP; bf16 similar).

First-run latency includes JIT compilation of Triton/Inductor kernels; caches
are isolated under `G:\zluda\.triton_cache` and `G:\zluda\.inductor_cache`.

## Native ROCm 10.1 on gfx1010 — what worked / what did not

Worked:
- HIP runtime, native `torch 2.14+rocm10.1`, ComfyUI;
- rocBLAS / rocFFT / rocSPARSE / MIOpen gfx1010 kernels;
- SDPA math backend.

Not available / failed:
- **hipBLASLt**: no gfx1010 kernels (library ships gfx11xx/gfx12xx only).
  PyTorch `linear`/`addmm` cuBLASLt path fails with
  `CUBLAS_STATUS_ALLOC_FAILED` unless `DISABLE_ADDMM_CUDA_LT=1`.
- **SDPA flash / memory-efficient**: “not compiled for current AMD GPU
  architecture” — kernels exist only for MFMA parts (gfx90a/gfx942/gfx950).
- **torch.compile**: native ROCm torch build has no Triton installation.
- **xformers**: no gfx1010 build (CUTLASS sm80 kernels; no RDNA1 support).
- **SageAttention**: no gfx1010 wheel; CUDA kernels target sm80+, fp8 paths need
  sm89.
- Stock rocBLAS: fallback Tensile kernels only (4.2-4.85 TF fp32). The tuned
  libraries must be supplied externally and match the shipped `rocblas.dll`
  from the same release; both are replaced together.

## ZLUDA — what works / what does not

Works: torch eager (fp32/fp16/bf16), cuBLAS→rocBLAS, cuDNN→MIOpen, cuFFT,
cuSPARSE, NVML; AMP training; torch.compile+Inductor (Triton); custom Triton
kernels; Triton flash attention forward+backward; FlexAttention; fp16
accumulation mode; cuBLASLt path through the new rocBLAS fallback (bias
included).

Does not work: PyTorch SDPA `flash`/`mem_efficient` (kernels are compiled only
for sm80+ and are absent from the cu118 PTX that ZLUDA consumes);
`xformers`/`flash-attn`/`sageattention` packages (no Windows wheels for
torch 2.7/cu118; their CUDA kernels are sm80+ SASS/PTX); TF32 MMA in Triton
(use fp16/bf16 inputs or `input_precision="ieee"`).

## Reproduce

- `run-zluda.cmd <app>` - ROCm 10.1 + tuned rocBLAS 7.1.1.
- venv: `G:\zluda\venv` (torch 2.7.1+cu118, triton-windows 3.3.1).
- Scripts: `bench/matmul_bench.py`, `bench/fp16_acc.py`,
  `bench/triton_flash_canon.py`, `bench/math_attn_fb.py`, `smoke/*`.
- Patch: `local-patches/zluda-rdna1-gfx1010.patch`.

## Stage 0 - roofline and regression baseline (2026-11-24)

Commands:

```bat
run-zluda.cmd venv\Scripts\python.exe bench\roofline.py
run-zluda.cmd venv\Scripts\python.exe bench\regression.py
```

| metric | result | note |
|---|---|---|
| FP32 ALU (Triton, 4 chains) | 5.0 TF / 51% | Triton codegen limit; rocBLAS GEMM reaches 8.16 TF |
| FP16 ALU (Triton, packed) | 18.8 TF / 96% | reaches the packed FP16 peak |
| copy / add / triad | 373-378 GB/s / 83-85% | 448 GB/s theoretical |
| reduction | 341-373 GB/s / 76-83% | read-only traffic |
| HBM latency | 172 ns/load | dependent pointer chase, 32 MB |
| cache-resident load | ~3.1 TB/s | 32 KB working set |
| clocks / power | unavailable | ZLUDA NVML returns NotSupported |

Regression gate: 12/12 correctness checks pass (matmul fp32/fp16/bf16,
linear, conv2d, softmax, layernorm, SDPA math). Reference 4096 GEMM:
fp32 8.16 TF, fp16 6.91 TF, bf16 4.40 TF; math SDPA causal 1.14 TF at 2048.

Clock/power monitoring is still open: neither ZLUDA NVML nor the installed
ROCm 10.1 user space exposes rocm-smi/amd-smi on this Windows host.

## Stage 1 - GEMV, batched GEMM and bf16 (2026-11-24)

### GEMV / skinny GEMM (N=K=4096)

`bench/gemv.py`, best of two runs:

| dtype | m | Triton | rocBLAS `F.linear` | % of 448 GB/s |
|---|---|---|---|---|
| fp16 | 1 | 0.114 ms / 294 GB/s | 0.296 ms / 114 GB/s | 66% |
| fp16 | 4 | 0.121 ms / 278 GB/s | 0.315 ms / 107 GB/s | 62% |
| fp16 | 8 | 0.139 ms / 242 GB/s | 0.309 ms / 109 GB/s | 54% |
| fp16 | 16 | 0.218 ms / 155 GB/s | 0.312 ms / 108 GB/s | 35% |
| fp16 | 32 | 0.435 ms / 78 GB/s | 0.447 ms / 76 GB/s | 17% |
| fp32 | 1 | 0.216 ms / 311 GB/s | 0.433 ms / 155 GB/s | 69% |
| fp32 | 4 | 0.236 ms / 284 GB/s | 0.453 ms / 148 GB/s | 63% |
| fp32 | 8 | 0.230 ms / 292 GB/s | 0.446 ms / 151 GB/s | 65% |
| fp32 | 16 | 0.415 ms / 163 GB/s | 0.456 ms / 148 GB/s | 36% |
| fp32 | 32 | 0.925 ms / 73 GB/s | 0.604 ms / 112 GB/s | 16% |

The Triton kernel loads each weight row once, keeps X in cache, uses fp16
products with blockwise fp32 accumulation for M<=16 and full fp32 products for
M=32. M=1/4/8 reach 54-69% of theoretical bandwidth, i.e. 78-82% of the
measured 377 GB/s copy ceiling. M=16/32 are reduction/compute-bound; for fp32
M=32 `F.linear` is faster and should be used as a fallback. `cublasSgemv`
(fp32) is unsupported by libcublas.so.

### GEMM shape grid and batched GEMM

`bench/stage1.py`:

| shape | fp32 | fp16 | bf16 |
|---|---|---|---|
| 2048^3 | 8.39 TF | 6.64 TF | 4.26 TF |
| 4096^3 | 8.16 TF | 6.92 TF | 4.40 TF |
| batched b=8, 1024^3 | 7.52 TF | 7.05 TF | - |
| strided-batched b=8, 1024^3 | 7.51 TF | 6.90 TF | - |

Tuned rocBLAS 7.1.1 is already shipped; the fp32 target of 8.8-9.2 TF and the
HH 14-16 TF target require new Tensile kernels and remain open. Batched and
strided-batched paths inherit the GEMM performance.

### Epilogues

64x4096x4096, fp32 / fp16:

| path | fp32 | fp16 |
|---|---|---|
| GEMM | 0.827 ms | 0.598 ms |
| linear + bias | 0.877 ms | 0.645 ms |
| GEMM + separate bias add | 0.858 ms | 0.636 ms |
| linear + bias + residual | 0.907 ms | 0.676 ms |
| linear + bias + GELU | 0.883 ms | 0.630 ms |

The cublasLt bias epilogue is correct but not faster than a separate add at
this size; residual/activation epilogues are unfused. Bias error matches the
unfused path (fp32 0, fp16 0.0038).

### bf16 emulation

`zluda_ops/bf16.py` converts bf16 operands to fp32, runs the tuned fp32 GEMM
and casts the result back:

| n | native bf16 | fp32 emulation | relative error |
|---|---|---|---|
| 2048 | 3.75 ms / 4.58 TF | 2.52 ms / 6.83 TF | 0.0139 (both) |
| 4096 | 31.23 ms / 4.40 TF | 17.35 ms / 7.92 TF | 0.0196 (both) |

This meets the 6-7 TF bf16 target at the cost of temporary fp32 tensors.

### int8

`torch._int_mm` is rejected by the cuBLASLt fallback with
`CUBLAS_STATUS_INVALID_VALUE`; no int8 path is available through ZLUDA today.
Quantization remains Stage 3 work.
## Stage 2 - SDPA-compatible Triton flash attention (2026-11-24)

`zluda_ops/flash_attention.py` packages the canonical fwd+bwd kernels as an
autograd Function with an SDPA-compatible entry point, a `FlashAttention`
module and a `flash_attn_func` shim. GQA/MQA is handled by expanding K/V
heads; unsupported cases (non-causal backward, masks, dropout, head_dim != 64,
sequence not divisible by 128) fall back to PyTorch math SDPA.

| case (fp16, B1 H8 D64) | Triton fwd | math fwd | fwd+bwd | fwd err | grad err |
|---|---|---|---|---|---|
| 512 causal | 0.48 ms | 0.84 ms | 1.38 ms | 0.002 | 0.002 |
| 1024 causal | 1.10 ms | 2.05 ms | 3.59 ms | 0.002 | 0.004 |
| 2048 causal | 3.11 ms | 7.54 ms | ~10.9 ms | 0.001 | 0.004 |
| 512 causal, GQA H=8/KV=2 | 0.54 ms | 0.85 ms | 1.59 ms | 0.002 | 0.006 |

Non-causal Triton forward matches math but the backward kernel is causal-only;
the wrapper uses math for non-causal training. Sliding window, varlen, softcap,
dropout and flash-decoding remain open (Stage 2.2/2.3).
## Out-of-the-box auto integration (2026-11-24)

`zluda_ops.auto` patches `F.linear` and `F.scaled_dot_product_attention` with
safe conditions and fallback. Scripts use `run-zluda-auto.cmd`; ComfyUI uses
the `zluda_ops_auto` custom node installed by `tools/install_auto.py`.

| case | original | auto | speedup | max abs err |
|---|---|---|---|---|
| linear fp16 M=1, K=4096 | 0.292 ms | 0.150 ms | 1.95x | 0.00024 |
| linear fp16 M=4, K=4096 | 0.309 ms | 0.150 ms | 2.06x | 0.00024 |
| linear fp16 M=8, K=4096 | 0.289 ms | 0.199 ms | 1.45x | 0.00049 |
| linear fp32 M=1, K=4096 | 0.459 ms | 0.219 ms | 2.10x | 0 |
| linear fp32 M=8, K=4096 | 0.447 ms | 0.365 ms | 1.22x | 0 |
| linear fp16 M=1, K=3200 | 0.193 ms | 0.118 ms | 1.63x | 0.00024 |
| bf16 linear 64x1024x1024 | 0.272 ms | 0.165 ms | 1.65x | 0.002 |
| bf16 linear 64x4096x4096 | 2.227 ms | 1.243 ms | 1.79x | 0.016 |
| causal SDPA 512, D64 | 0.657 ms | 0.413 ms | 1.59x | 0.001 |

Auto GEMV uses fp32 products and fp32 accumulation; bf16 auto converts to
fp32 and casts back; FA keeps fp32 softmax/LSE/accumulation. No int16 path
exists and int8 remains unsupported. Details: `INTEGRATION.md`.
## P0 training stability (2026-11-24)

The `bench_zluda_model_compile.py` failure was reproduced: PyTorch selected a
cuDNN/CUTLASS FMHA kernel (`fmha_cutlassF_f16_aligned_64x64_rf_sm80`) that
cannot execute on gfx1010; after the failed kernel attempts, backward reported
`cublasSetWorkspace` `<unknown>`.

Workaround/fix:
- `zluda_ops/mha.py` forces the slow math path for `nn.MultiheadAttention`
  (`need_weights=True`, fastpath disabled) and excludes it from Dynamo.
- `run-zluda-train.cmd` installs the MHA patch, disables inference
  auto-dispatch and caps VRAM at 85%.

Result on the user model (`FeatureSetRanker`, fp16 autocast, 8 layers):
- eager full step: **1040–1208 ms**, loss **0.00683**;
- **0 FATAL** lines, no `cublasSetWorkspace` error;
- `torch.compile` is skipped for this model: ~30 minutes even on an NVIDIA T4; not a P0 blocker.
## P0 fp16 accumulation UX (2026-11-24)

`zluda_ops.fp16acc` exposes a warning-emitting context manager and an explicit
unsafe opt-in guard. Default remains fp32 accumulation.

| n | fp32 accum | fp16 accum | max abs diff | relative max |
|---|---|---|---|---|
| 1024 | 0.40 ms / 5.36 TF | 0.20 ms / 10.98 TF | 1.78 | 0.0109 |
| 2048 | 2.57 ms / 6.69 TF | 1.76 ms / 9.77 TF | 4.63 | 0.0206 |
| 4096 | 19.81 ms / 6.94 TF | 11.14 ms / 12.34 TF | 8.00 | 0.0214 |

## P0 monitoring (2026-11-24)

`bench/monitor.py` samples Windows GPU utilization and VRAM (idle: ~1.6%,
8176 MB visible). ZLUDA NVML reports `NotSupported` for utilization,
temperature, clocks and power; no rocm-smi/amd-smi is bundled with the Windows
ROCm 10.1 user space, so clocks/power/temperature remain unavailable.

## P0 non-causal flash attention (2026-11-24)

The Triton backward kernel now has a non-causal path. B1 H8 D64 fp16:

| s | fwd | fwd+bwd | fwd err | grad err |
|---|---|---|---|---|
| 512 | 0.39 ms (math 0.74) | 1.63 ms | 0.0002 | 0.0005 |
| 1024 | 1.12 ms (math 2.15) | 3.66 ms | 0.0002 | 0.0004 |

Masks, dropout, sliding window, varlen and flash-decoding still use the math
fallback or remain open.
## P0 sliding window / varlen (2026-11-24)

`flash_attn_func(window_size=...)` and `flash_attn_varlen_func` are available.
They currently use the correct math fallback (autograd verified); the Triton
kernel still handles only mask/dropout-free causal/non-causal attention.

| case | result |
|---|---|
| sliding causal 128, window (32,0) | fwd max err 0 |
| varlen [64,128], causal | fwd max err 0.001, grad OK |
| non-causal 512/1024 Triton | fwd err 2e-4, grad err 5e-4/2e-4 |

## P0 auto GEMV M<=32 (2026-11-24)

`zluda_ops.auto` dispatches Triton GEMV for M<=16 fp32 and M<=32 fp16
(inference/no-grad only). Correctness vs original `F.linear`: fp16 max err
5e-4, fp32 <=1.4e-6. M=16/32 target bandwidth (70-80%) remains open; fp32 M=32
uses the rocBLAS fallback because it is faster.