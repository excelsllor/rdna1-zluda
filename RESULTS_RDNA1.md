# ZLUDA on Radeon RX 5700 XT (gfx1010 / RDNA1) — build and benchmark results

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
