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

Measurements use random data. Zeros/ones/uninitialized buffers can be ~1 TF
higher because of GPU power/compression effects and are not representative.

| path | fp32 | fp16 (fp32 accum) | fp16 (fp16 accum) | bf16 native | bf16 fp32 emulation |
|---|---|---|---|---|---|
| ROCm 10.1 stock rocBLAS (fallback Tensile) | 4.85 TF | 4.21 TF | — | 2.32 TF | — |
| **ROCm 10.1 + tuned rocBLAS 7.1.1** | **8.16 TF** | **6.94 TF** | **12.1 TF default / 16.73 TF override** | **4.42 TF** | **7.92 TF** |
| native ROCm 10.1 torch | 4.19 TF | 3.25 TF | — | 2.38 TF | — |

HH 2048: **17.94 TF** with the existing-kernel override (default 9.5 TF).
SS/HPA targets remain open (8.16/7.19 TF).
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
| fp16 | 16 | 0.192 ms / 176 GB/s | 0.276 ms / 122 GB/s | 39% |
| fp16 | 32 | 0.410 ms / 83 GB/s | 0.408 ms / 84 GB/s | 19% |
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
## P0 edge cases: masks, dropout, sliding, varlen (2026-11-24)

`scaled_dot_product_attention`/`flash_attn_func`/`flash_attn_varlen_func`
accept masks, dropout, sliding `window_size` and packed varlen inputs. These
paths currently use the correct math fallback; fwd/bwd completed without
errors (mask grad ~2e-5, dropout grad ~1.4e-5).

| case | result |
|---|---|
| bool mask 128x128 | fwd+backward OK |
| dropout_p=0.1 | fwd+backward OK |
| sliding causal window (32,0) | fwd err 0 |
| varlen [64,128], causal | fwd err 0.001, grad OK |

## P0 compile smoke (2026-11-24)

A single `nn.TransformerEncoderLayer` (d=128, 4 heads, seq 32) compiled with
`torch.compile(dynamic=True, fullgraph=False)` and ran in ~1 s with the
slow-path MHA. The full 8-layer model compile remains skipped (~30 min even on
an NVIDIA T4).

## P0 Tensile gap (2026-11-24)

No Tensile toolchain or rocblas-bench is present in the Windows ROCm 10.1
package; only the existing tuned rocBLAS 7.1.1 libraries exist. The fp32
8.8-9.2 TF and HH 14-16 TF targets require new `.dat/.co` kernels and remain
blocked externally.
## P0-5 Tensile kernels already present (2026-11-24)

The tuned rocBLAS 7.1.1 library contains 110 HH solutions for gfx1010. A
`ROCBLAS_TENSILE_GEMM_OVERRIDE_PATH` profile selects better existing kernels;
no Tensile rebuild was needed.

| n | default HH | override HH | speedup |
|---|---|---|---|
| 1024 | ~12.4 TF | ~12.9 TF | 1.04x |
| 2048 | ~9.5 TF | **17.94 TF** | 1.89x |
| 4096 | ~12.1 TF | **16.73 TF** | 1.38x |

Profile: `tensile_overrides/gfx1010_hh.csv`; launcher: `run-zluda.cmd`.
Raw index mapping on this build: `raw = -solution_index - 10`.

SS remains ~8.16 TF and HPA ~7.19 TF; their P0-5 targets are not reached by
the existing kernels. Measurements with zeros/ones/uninitialized buffers are
inflated (SS up to ~9.1 TF) and are not representative of random model data.
## P0-1 fp32 GEMM "gap" between direct rocBLAS and PyTorch (2026-11-25)

Earlier notes claimed 9.2-9.6 TF from isolated `rocblas_sgemm`/`rocblas_gemm_ex`
scripts versus 8.16 TF for `torch.matmul`. `bench/rocblas_gap.py` reproduces
every variant in one warmed process (shuffled rounds, median of 5):

| variant (4096^2 fp32, random data unless noted) | median | TF |
|---|---|---|
| `torch.mm(a, b)` | 16.79 ms | 8.19 |
| `torch.mm(a, b, out=c)` | 16.80 ms | 8.18 |
| direct `rocblas_sgemm`, torch buffers | 16.79 ms | 8.19 |
| direct `rocblas_sgemm`, hipMalloc buffers | 16.80 ms | 8.18 |
| direct `rocblas_sgemm`, **zeroed** hipMalloc buffers | **14.90 ms** | **9.23** |

The direct call is bit-identical to `torch.mm` (max abs error 0). The only
difference that matters is the data: zero/uninitialised buffers run ~12.8%
faster because the GPU draws less power and holds higher clocks. The old
raw-HIP scripts allocated `hipMalloc` buffers without initialising them, so
their 9.2-9.6 TF numbers were power artifacts, not a software gap.

Consequences:

- fp32 4096^2 on random data is **8.16-8.20 TF = 84% of the 9.754 TF peak**;
  this is the practical ceiling of the shipped Tensile kernels. The 8.8-9.2 TF
  target is withdrawn.
- Every benchmark must use random data and median timing; report format is in
  `bench/common.py` and `bench/results/rocblas_gap-*.json`.
- HPA behaves the same way: 6.94 TF random versus ~7.19 TF on zeroed data.

## P0-5 clocks/power monitoring and measurement protocol (2026-11-25)

Clocks remain unavailable, now with a concrete root cause instead of "no tool":

| probe | result |
|---|---|
| ZLUDA NVML (`torch.cuda.clock_rate`, `temperature`, `power_draw`) | `NotSupported` |
| `rocm-smi`/`amd-smi` in ROCm 10.1 Windows user space | not shipped |
| `atiadlxx.dll` (Adrenalin driver) loaded, adapters enumerated | `AMD Radeon RX 5700 XT` visible |
| `ADL2_OverdriveN_PerformanceStatus_Get` | `ADL_ERR_NOT_SUPPORTED` (-8) |
| `ADL2_Overdrive6_CurrentStatus_Get` / `Temperature_Get` | `ADL_ERR_NOT_SUPPORTED` (-8) |
| `ADL_Overdrive5_CurrentActivity_Get` | `ADL_ERR` (-1) |
| GPU utilization / VRAM | available (Windows performance counters) |

`bench/gpu_clocks.py` implements the ADL probe and degrades to null values with
an error string, so reports stay valid.

To keep results comparable without clock telemetry, the protocol is now part of
every report (`bench/common.py:save_report`): random data only, warmup per
variant, mean per round, **median across rounds**, variant order shuffled when
comparing, and the limits are recorded explicitly. `bench_median` is available
for new benchmarks. This is why the P0-1 zero-data artifact and earlier +-10%
run-to-run differences are no longer mixed into the numbers.

## P0-2 Triton flash-attention masks (2026-11-25)

Attention masks now run inside the Triton fwd/bwd kernels instead of the math
fallback. Supported: boolean masks (2D, 3D per batch, 4D per batch+head, with
broadcast), additive float masks, and masks combined with causal; the kernel
keeps broadcast masks as stride-0 views, so no dense mask is materialised.
Masked lanes are zeroed after the exponential (the old `-1e6` trick would leak
into long rows), and dropout/window/varlen still use math SDPA.

`bench/fa_mask.py` (B1 H8 D64 fp16, s=512):

| case | fwd err | grad err |
|---|---|---|
| bool 2D | 2.4e-4 | 9.8e-4 |
| additive 2D | 4.9e-4 | 9.8e-4 |
| bool 4D per head | 2.4e-4 | 9.8e-4 |
| additive 3D per batch | 4.9e-4 | 9.8e-4 |
| bool 2D + causal | 9.8e-4 | 2.0e-3 |
| additive 2D + causal | 9.8e-4 | 3.9e-3 |

Speed (s=1024, bool mask, median of shuffled rounds): masked causal Triton
**1.22 ms vs 2.04 ms math = 1.67x**; non-causal 1.82 vs 1.98 ms = 1.09x.

## P0-2b Triton flash-attention dropout (2026-11-25)

Dropout now runs in the Triton kernels: inverted dropout on the attention
weights, with the row normalization using the sum of the *original*
probabilities, so the output matches torch SDPA semantics (E[out] equals the
non-dropout output). The same Philox mask is regenerated in backward from the
saved seed and per-(head, query, key) offsets, so no dropout mask is stored.

`bench/fa_dropout.py` (B1 H8 D64 fp16, s=256):

| check | result |
|---|---|
| dropout=0 vs math SDPA | fwd err 2.0e-3, grad err 3.9e-3 (unchanged) |
| unbiasedness p=0.1 / p=0.5, 16 seeds | z = 0.82 / 0.81 (unbiased, threshold 3) |
| determinism | same seed identical, different seed differs |
| gradients | finite, mean abs comparable to no-dropout |
| s=1024 causal, p=0.1 | **1.04 ms vs 1.97 ms math = 1.90x** |

## P0-3 Triton sliding window and varlen (2026-11-25)

The sliding window is now arithmetic inside the Triton fwd+bwd kernels: block
iteration starts at `max(0, block_start - left)` aligned down to BLOCK_N and
skips blocks beyond the right bound; per-element window bounds are applied
without allocating a dense mask. A dense-mask fallback remains only for
unsupported shapes.

| case (s=512, B1 H8 D64 fp16) | fwd err | grad err |
|---|---|---|
| causal window (32,0) | 2.0e-3 | 2.0e-3 |
| causal window (64,0) | 2.0e-3 | 2.9e-3 |
| non-causal window (16,16) | 1.0e-3 | 2.0e-3 |
| non-causal window (64,32) | 4.9e-4 | 1.1e-3 |
| varlen [128,256,384] causal window (64,0) | 1.0e-3 | 3.9e-3 |

Speed (s=1024): window (128,0) causal **0.26 ms vs 1.87 ms** math with dense
mask (7.1x); non-causal window (64,0) 0.26 vs 1.85 ms (7.15x).

`flash_attn_varlen_func` now dispatches per sequence through the Triton kernel
without dense masks; a fused packed-varlen kernel is still open. The backward
preprocess kernel assumes a contiguous (B,H,N,D) layout, so the Triton path now
makes inputs contiguous (a no-op for normal calls) — this also fixes varlen
gradients through transposed views.

## P0-4 GEMV M=16/32 (2026-11-25)

The M=16/32 target (70-80% of 448 GB/s) is not reachable with the reduction
patterns available on gfx1010. Measured at N=K=4096, fp16 weights:

| approach | M=16 | M=32 |
|---|---|---|
| row-per-program kernel (`_gemv_mvec`), best config | **176.5 GB/s (39%)** | **83.1 GB/s (19%)** |
| split-K + `tl.dot`, best of 6 configs | 81.5 | 80.5 |
| packed-fp16 products (PROD_F16) | 151.7 | 31-50 |
| fp32 rocBLAS (`F.linear`) reference | 159.6 | 118.6 |

`bench/gemv_m32.py` implements the split-K/`tl.dot` variant and the config sweep.
The `tl.dot` path is slower because small M tiles do not provide enough
parallelism and the atomic reduction adds traffic; packed-fp16 products reduce
precision and throughput at these shapes.

Decisions taken:

- `bench/gemv.py` now uses fp32 products for M=16 (better accuracy and
  **+13%** throughput, 176 vs 155 GB/s); peak numbers for M=1 improved to
  302/331 GB/s (fp16/fp32) with the measurement protocol from P0-5.
- M=32 keeps the rocBLAS fallback on the auto path (119 GB/s fp32 vs 76-83
  Triton), and the 70-80% target is recorded as unreachable rather than open.

## P1-1 Winograd / MIOpen convolution audit (2026-11-25)

`bench/conv.py` measures 3x3 stride-1 conv2d on ResNet/SD shapes, checks
correctness against CPU and captures the MIOpen solver choice. Correctness:
max abs 2.5e-5 (relative 4.8e-7) vs CPU for fp32.

The MIOpen logs (MIOPEN_ENABLE_LOGGING=1, MIOPEN_LOG_LEVEL=6) show that for
NCHW fp16 3x3 the selected solution is **`GemmFwdRest` with
`miopenConvolutionFwdAlgoGEMM`** (find time 0.1052 ms): MIOpen lowers the
convolution to a rocBLAS GEMM. Winograd and HIP implicit-GEMM solvers are not
selected for gfx1010 here. Additional environment findings:

- `MIOpen: Warning [ParseAndLoadDb] File is unreadable: gfx1010_20.HIP.fdb.txt`
  — there is no gfx1010 perf-db, so every new shape runs a runtime find.
- `MIOpen: Warning [OpenRuntimeLibraryForDevice] CK grouped conv library not
  found for device gfx1010` — the CK grouped-conv path is unavailable.

Performance (NCHW, s=forward, 3x3 s1 p1):

| shape (B,C,H,W,O) | fp32 fwd | fp16 fwd | fp32 fwd+bwd | fp16 fwd+bwd |
|---|---|---|---|---|
| 1x64x56x56x64 | 0.08 ms | 0.16 ms | 0.48 ms | 0.67 ms |
| 8x64x56x56x64 | 0.22 ms | 0.72 ms | 1.16 ms | 2.82 ms |
| 8x256x14x14x256 | 0.18 ms | 0.96 ms | 0.75 ms | 2.39 ms |
| 2x256x64x64x256 | 0.67 ms | 2.14 ms | 2.92 ms | 6.24 ms |

Two practical results:

- **channels_last is unusable** on gfx1010: 10-1000x slower than NCHW
  (e.g. 2x256x64x64 fp32: 546 ms vs 0.67 ms). Keep NCHW.
- **fp16 convolution is 2-4x slower than fp32** through MIOpen on this card,
  even though the selected solver is a GEMM in both cases. This matters for
  fp16 SD/ComfyUI workloads and is a candidate follow-up (e.g. running
  selected convs in fp32 or forcing a different solver).

Decision for the roadmap: no custom Winograd path. MIOpen already lets its
find choose among solvers (ZLUDA calls `miopenFindSolutions` and takes the
fastest), and for gfx1010 it picks the GEMM fallback; Winograd is "not
beneficial/not selected" rather than an opportunity. The remaining conv win
would come from the missing gfx1010 perf-db and from the fp16-vs-fp32 anomaly.

## P1-2 Flash-decoding (split-KV) for decode (2026-11-25)

`zluda_ops/flash_decode.py` implements decode attention as split-KV flash
decoding: each (batch, head) is split into up to 16 KV chunks, a Triton kernel
computes partial (acc, m, l), and a merge kernel combines the partials with the
standard log-sum-exp formula. Reference is float32 scores/softmax; the M query
tokens are assumed to be the last M positions when causal.

| case (B1 H8 D64 fp16) | max rel err | flash-decoding | math SDPA | speedup |
|---|---|---|---|---|
| M=1, N=1024 | 4.6e-4 | 0.090 ms | 0.306 ms | **3.39x** |
| M=1, N=2048 | 4.6e-4 | 0.107 ms | 0.357 ms | **3.33x** |
| M=1, N=4096 | 3.6e-4 | 0.131 ms | 0.465 ms | **3.55x** |
| M=8 casual, N=4096 | 2.9e-4 | 0.137 ms | 0.615 ms | **4.50x** |
| M=8, N=2048 | 4.6e-4 | 0.193 ms | 0.360 ms | 1.86x |
| M=4, N=4096 | 4.5e-4 | 0.133 ms | 0.467 ms | 3.52x |

The milestone target (>=1.5x decode at 4k) is met. `zluda_ops.auto` now
dispatches `F.scaled_dot_product_attention` decode calls (fp16, head_dim 64,
M<=16, KV>=512, no mask/dropout/grad, default scale) to flash-decoding: output
matches the direct kernel bit-exactly (max abs 2.8e-5 vs float32 reference) and
is 1.33x faster than math through the patched Python entry point (the raw kernel
is 3.35x; the difference is Python dispatch overhead at these small latencies).

`bench/flash_decode.py` reproduces both the correctness checks and the timings.

## P1-3 Weight-only int4/int8 quantization (2026-11-25)

`zluda_ops/quant.py` adds groupwise symmetric weight-only storage
(groups 32/64/128, fp16 scales, two signed nibbles per byte) with int8/int4
dequantizers and a fused Triton int4 dequant-GEMV that reads packed weights
directly (no fp16 weight materialisation). The storage layout is documented and
GPTQ/AWQ-style (groupwise scale); their exact nibble order/zero-point loading is
a follow-up.

| format | relative RMS error (random normal) | compression | 7B weights |
|---|---|---|---|
| int4, group 32 | 9.7% | 3.56x | 3.5 GB |
| int4, group 64 | 10.8% | 3.76x | 3.5 GB |
| int4, group 128 | 11.7% | 3.88x | 3.5 GB |
| int8, group 128 | 0.65% | 1.97x | 7.0 GB |

The int4 rounding error is that of plain RTN (no GPTQ error compensation); on
real models GPTQ/AWQ-style compensation is expected to reduce it.

Fused int4 GEMV (N=K=4096, fp16 activations):

| M | int4 GEMV | fp16 GEMV | torch (dequant+linear) | speedup vs fp16 GEMV | max abs vs fp32 ref |
|---|---|---|---|---|---|
| 1 | 0.078 ms (114 GB/s packed) | 0.113 ms (297 GB/s) | 0.272 ms | **1.44-1.63x** | 1.6e-4 |
| 8 | 0.202 ms | 0.180 ms | 0.274 ms | 0.89x | 2.0e-4 |

Correctness of the fused kernel vs the fp32 dequant+linear reference is within
2.4e-4 for M=1..16. The 70-80% of 448 GB/s target is not met: at 0.5 byte/weight
the dequantisation (unpack, scale, converts) is compute/launch bound rather than
bandwidth bound. Fixing register spills (smaller BK for M>=4 and contiguous
paired x loads) took M=8 from 0.84 ms to 0.20 ms. Next steps: tiled/split-K int4
GEMV, exact GPTQ/AWQ loading, and the prefill dequant-GEMM.

`bench/quant.py` reproduces the error, correctness and speed tables.
### P1-3 follow-up: tuning and hypothesis ablation (2026-11-25)

Three suspected bottlenecks were tested in isolation (warmed GPU, shuffled
interleaved rounds, medians; note that raw per-process variance is still
+-15-30% because the GPU clocks cannot be fixed):

| variant (M=1, K=4096, group 64) | time | conclusion |
|---|---|---|
| per-chunk `tl.sum` + broadcast scales (adopted) | 0.118 ms | baseline |
| per-byte scale gather (old) | 0.135 ms | **H1 confirmed**: broadcast scales are ~12-14% faster |
| fp32 products/conversions | 0.116 ms | **H3 refuted**: fp16 products are not faster; conversions are not the bottleneck |
| vector accumulator, one final sum | 0.134 ms | **H2 partially refuted**: fewer calls to `tl.sum` is slower (register spills) |
| memory-only floor (no nibble unpack) | 0.105 ms | only ~11% above the floor, so the kernel is close to its structural limit |
| tiled multi-row kernel (BN=4, BK=2048) | 0.083 ms | ~6% better than the reference in the same process; M=8 is much worse (register pressure) |
| split-K with atomic add | worse | atomic reduction costs more than the extra parallelism |

Final tuned configuration: M=1 uses BK=4096 with fp16 products + fp32
accumulation (0.062 ms, 145 GB/s packed, **1.92x** vs the fp16 GEMV); M=8 uses
BK=1024 (0.200 ms, 0.93x). Fully fp16 accumulation (products + per-chunk sum in
fp16) runs at the same speed and is less accurate, so it is not used. The
remaining gap to the 70-80% packed-bandwidth target is structural: one reduction
per output row with only 0.5 byte/weight of traffic; a multi-row tiling is the
only measured improvement (+6%) and is not worth the complexity for now.

## P1-5 Fused transformer kernels (2026-11-25)

`zluda_ops/fused.py` implements Triton kernels for the pointwise/rowwise ops
that TorchInductor does not fuse well on RDNA1:

* RMSNorm fwd+bwd (one-pass row kernel; weight gradient via atomic add);
* LLaMA-style rotate-half RoPE fwd+bwd;
* fused SiLU(gate)*up (SwiGLU) fwd+bwd;
* row softmax;
* chunked cross-entropy that never materialises the full (N, V) log-softmax.

Measured against torch eager on RDNA1 (fp16, VRAM 85%):

| op | shape | fused | torch eager | speedup |
|---|---|---|---|---|
| RMSNorm fwd | 8192x4096 | 0.52 ms | 4.29 ms | **8.2x** |
| RMSNorm fwd+bwd | 8192x4096 | 7.25 ms | 44.4 ms | **6.1x** |
| softmax | 8192x4096 | 1.27 ms | 23.5 ms | **18.5x** |
| RoPE fwd | 1x32x2048x128 | 0.30 ms | 1.54 ms | **5.1x** |
| RoPE fwd+bwd | 1x32x2048x128 | 0.47 ms | 4.13 ms | **8.8x** |
| SwiGLU fwd | 8192x2048 | 0.54 ms | 0.66 ms | **1.2x** |
| SwiGLU fwd+bwd | 8192x2048 | 1.11 ms | 1.71 ms | **1.5x** |
| chunked CE | 4096x16384 | 38 ms / 736 MB | 16.5 ms / 1024 MB | 0.4x (memory **-28%**) |

Numerical errors are at the fp16 accumulation level (RMSNorm 7.8e-3 forward on
unit-scale rows; gradients agree with torch within fp16 tolerance). The torch
baselines for RMSNorm/softmax are unfused fp32-upcast paths, which is exactly
the gap these kernels target; TorchInductor (`torch.compile`) was not used as a
baseline here because full-model compile is skipped on this stack (~30 min even
on an NVIDIA T4). Chunked CE is kept as a memory-only option: it saves 28% peak
memory at 4096x16384 but is slower than torch's fused CE.

LayerNorm, GeGLU, fused AdamW/clip and GPU sampling remain open (see ROADMAP).

## P1-1 follow-up: fp16 conv anomaly and the cast-to-fp32 workaround (2026-11-25)

Root cause of the 2-4x fp16 conv slowdown: MIOpen selects **different solver
families per dtype**. For fp32 3x3 it picks `ConvBinWinogradRxSf2x3g1`
(miopenConvolutionFwdAlgoWinograd, find time 0.065 ms); for fp16 the same
shape falls back to `GemmFwdRest` (miopenConvolutionFwdAlgoGEMM). Winograd is
not selected for fp16 on gfx1010.

Workaround implemented in `zluda_ops/conv.py`: for eligible fp16 3x3 stride-1
dilation-1 group-1 convolutions, cast inputs and weights to fp32, run the fp32
Winograd conv and cast the result back. fp16 inputs are exactly representable in
fp32, so the result equals an fp32 convolution rounded to fp16 (gradients flow
through the casts). `zluda_ops.auto` patches `F.conv2d` with it when
`ZLUDA_CONV_FP32=1`.

Measured with the warmed/shuffled protocol (VRAM 85%):

| shape | fwd fp16 | fwd cast | fwd speedup | fwd+bwd fp16 | fwd+bwd cast | bwd speedup |
|---|---|---|---|---|---|---|
| 1x64x56x56, 64 | 0.158 ms | 0.112 ms | **1.41x** | 0.478 ms | 0.596 ms | 0.80x |
| 8x64x56x56, 64 | 0.687 ms | 0.289 ms | **2.37x** | 1.562 ms | 1.532 ms | 1.02x |
| 8x256x14x14, 256 | 1.013 ms | 0.239 ms | **4.25x** | 1.706 ms | 0.900 ms | **1.90x** |
| 2x256x64x64, 256 | 2.187 ms | 0.861 ms | **2.54x** | 4.490 ms | 3.471 ms | **1.29x** |
| 1x320x64x64, 320 | 1.573 ms | 0.690 ms | **2.28x** | 3.173 ms | 2.707 ms | **1.17x** |
| UNet block 1x256x32x32 | 0.430 ms | 0.250 ms | **1.72x** | - | - | - |
| UNet block 2x320x64x64 | 3.259 ms | 1.419 ms | **2.30x** | - | - | - |

Forward wins on 5/5 shapes (1.4-4.3x), forward+backward on 4/5 (up to 1.9x);
the smallest shape is conversion-bound for training (0.80x). Gradient error vs
the fp32 reference is 2e-8..7e-4 (fp16 accumulation level). `bench/conv_fp16.py`
reproduces the table.

Caveats: fp32 copies of the conv weights cost memory; the win is shape-dependent
(conversion cost for tiny convs); only 3x3 stride-1/dilation-1/group-1 is
eligible, everything else uses the regular path.

## MIOpen find-db and solver investigation (2026-11-25)

Follow-up to the conv audit: does gfx1010 need a MIOpen perf-db, and are the
auto-selected solvers suboptimal?

Findings (`bench/miopen_db.py`):

- The shared MIOpen DLL contains no gfx1010 strings, i.e. no system find-db for
  this architecture ships with the ROCm 10.1 user space
  (`gfx1010_20.HIP.fdb.txt` is absent, matching the earlier warning).
- A **user find-db works and is auto-populated**: `~/.miopen/db/...ufdb.txt` and
  `~/.miopen/cache/...ukdb` files exist for gfx1010 and their mtime advances
  after new shapes run. Per-shape find cost on a warm cache is ~4.5 ms.
- The ~1.2-1.3 s spent on the very first convolution in a fresh process is
  MIOpen kernel-db/context initialisation (a dummy 8x8x8 conv absorbs it), not
  a solver search; it cannot be removed with a perf-db, only amortised by
  keeping the process alive.
- `MIOPEN_FIND_MODE=FAST` (heuristic only) gives identical steady-state times
  on the SD/ResNet shapes tested (e.g. 0.210 vs 0.213 ms at 1x64x56x56).
- `MIOPEN_FIND_ENFORCE=SEARCH` (forced exhaustive search) costs 45-124 ms extra
  on the first call and finds the same solution: steady state is unchanged
  (2.14 vs 2.22 ms at 2x256x64x64). The cached solution is already the fastest.

Conclusion: no system perf-db work is needed for gfx1010; the auto-selected
solver is optimal for the measured shapes, the user find-db is already in use,
and the cold-start cost is process initialisation. The remaining conv win is
the dtype/solver split handled by the cast-to-fp32 workaround (previous
section), and possibly the missing channels_last path (measured unusable).

## Fused kernels extension: LayerNorm, GeGLU, fused AdamW (2026-11-25)

Added to `zluda_ops/fused.py` (see the earlier P1-5 section for RMSNorm/RoPE/
SwiGLU/softmax): exact-erf GeGLU fwd+bwd, LayerNorm fwd+bwd and an in-place
fused AdamW step. `bench/fused.py` covers all cases. Measured after a clean
idle-GPU window (fp16, VRAM 85%):

| op | shape | fused | torch eager | speedup |
|---|---|---|---|---|
| RMSNorm fwd | 8192x4096 | 0.402 ms | 3.787 ms | **9.4x** |
| RMSNorm fwd+bwd | 8192x4096 | 4.57 ms | 13.49 ms | **2.95x** |
| softmax | 8192x4096 | 0.371 ms | 0.870 ms | **2.3x** |
| RoPE fwd | 1x32x2048x128 | 0.132 ms | 1.433 ms | **10.9x** |
| RoPE fwd+bwd | 1x32x2048x128 | 0.396 ms | 3.282 ms | **8.3x** |
| SwiGLU fwd/bwd | 8192x2048 | 0.305/0.935 ms | 0.457/1.397 ms | 1.50x/1.49x |
| GeGLU fwd/bwd | 8192x2048 | 0.303/0.936 ms | 0.467/1.393 ms | 1.54x/1.49x |
| LayerNorm fwd | 8192x4096 | 0.405 ms | 0.480 ms | 1.19x |
| LayerNorm fwd+bwd | 8192x4096 | 4.10 ms | 2.15 ms | **0.53x (slower)** |
| AdamW step | 4 Mi elements | 0.331 ms | 1.130 ms | **3.4x**, param diff 3e-8 |
| chunked CE | 4096x16384 | 6.6 ms / 736 MB | 3.5 ms / 1024 MB | 0.52x (memory **-28%**) |

Notes:

- **LayerNorm**: the forward kernel wins slightly, but training is slower than
  torch's native LayerNorm (the atomic weight/bias gradients dominate), so use
  `torch.nn.LayerNorm` for training; the Triton version is useful only for
  inference-style forward-only paths.
- **Chunked CE** remains a memory-only option (-28% peak memory, slower).
- GeGLU uses `libdevice.erf` (exact GELU, matches `F.gelu`), not the tanh
  approximation.
- Numbers vary run-to-run on this card (clocks cannot be fixed); the table is
  from a verified idle-GPU window. During this session a concurrent external
  benchmark had to be stopped and the GPU/driver recovered before the final
  measurement — benchmarks must not share the GPU with other jobs.

## P1-3 extension: prefill dequant-GEMM and tile decision (2026-11-25)

`quant.int4_matmul` adds a fused int4-dequant GEMM (nibbles unpacked and scaled
inside each `tl.dot` tile, weights never materialised as fp16) for prefill
(M=32..256). `quant.int4_prefill_linear` is the dequantize-once reference path.
Measured at N=K=4096 fp16, VRAM 85% (warmed, shuffled rounds):

| M | fused int4 GEMM | dequant-once + GEMM | fp16 weights (roofline) | fused vs dequant | fused vs fp16 |
|---|---|---|---|---|---|
| 32 | 0.614 ms | 2.667 ms | 0.445 ms | **4.3x** | 0.72x |
| 64 | 0.616 ms | 2.846 ms | 0.618 ms | **4.6x** | 1.00x |
| 128 | 1.161 ms | 3.070 ms | 0.873 ms | **2.6x** | 0.75x |
| 256 | 2.059 ms | 3.788 ms | 1.563 ms | **1.8x** | 0.76x |
| M=1 (decode) | fused GEMV 0.069 ms | 2.525 ms | 0.312 ms | **37x** | **4.6x** |

Errors vs the float32 reference are 2-4e-4. Conclusions:

- **Decode** stays on the fused int4 GEMV: 4.6x faster than fp16 weights and
  37x faster than dequantize-per-call.
- **Prefill**: fused int4 GEMM is the best option when the weights must stay
  int4 (4x less VRAM): it is at parity with cached fp16 weights at M=64 and
  25-30% slower at M=128-256 (the dequant ALU cost is no longer hidden).
  Dequantizing per call into fp16 and then GEMM is 1.8-4.6x slower and is only
  a reference. Serving a model with a fp16 weight cache is faster but loses the
  int4 memory saving — the trade-off is now measured.
- **Tiled multi-row int4 GEMV** (measured in the P1-3 ablation) was ~6% faster
  than the tuned kernel in one process, which is inside the run-to-run variance
  (±15-30% because GPU clocks cannot be fixed), so it is **not adopted**; the
  simpler M=1 kernel is kept.

`bench/quant_prefill.py` reproduces the table.
