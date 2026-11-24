# Roadmap — ZLUDA on gfx1010 (Radeon RX 5700 XT)

Status: ZLUDA v7-preview.21 builds and runs on gfx1010 with a unified ROCm 10.1
runtime + tuned rocBLAS 7.1.1. See `RESULTS_RDNA1.md` for measurements.

## Done (2026-11-24)

- [x] Build upstream ZLUDA v7-preview.21 (LLVM static, x64 release).
- [x] gfx1010 fixes: `:xnack-` ISA/CPU parsing, `dp4a` software fallback,
      `ld`/`st` cache-hint parsing, llvm_zluda IR shim for
      `ldmatrix.m8n8.x2.trans.b16`.
- [x] cuBLASLt→rocBLAS fallback with per-row bias and fp16-compute descriptors.
- [x] Tuned rocBLAS for gfx1010: HIP 6.4.2 package and ROCm 10.1 + rocBLAS 7.1.1
      unified stack (fp32 8.18 TF, ~84% of peak; fp16 HPA 6.96 TF).
- [x] Optional fp16 accumulation path (packed `v_pk_fma_f16`): up to 12.7 TF.
- [x] PyTorch 2.7.1+cu118 venv, Triton 3.3.1, torch.compile/Inductor.
- [x] Triton flash attention forward+backward (correct gradients) and
      FlexAttention; benchmarks vs math SDPA.

## Next

### 1. Custom Tensile tuning for gfx1010 (GEMM)
- Build Tensile toolchain matching rocBLAS 7.1.1 (ROCm 7.1.1 sources).
- Generate/tune `SS` (fp32), `HH_HPA` (fp16 in, fp32 accum) and `HH`
  (fp16 in/out, fp16 accum) for the shapes used by ML workloads (e.g.
  1024/2048/4096 square, K in 1024..16384, batched).
- Targets: fp32 8.8–9.2 TF, HPA 7.5–8 TF, HH 14–16 TF.
- Effort: hours to days on this GPU; ship resulting `.dat`/`.co` sets and a
  launcher profile. Risk: medium (library format/version must match `rocblas.dll`).

### 2. Packed-fp16 inference GEMM
- Tune `HH` kernels for inference tile shapes; validate accuracy vs fp32
  accumulation; expose a deliberate opt-in (torch flag
  `_set_cublas_allow_fp16_accumulation`, or a ZLUDA env switch) and document it.
- Consider mixed strategies: fp32 accumulation for training, fp16 accumulation
  for inference only.

### 3. Attention integration
- Wrap the Triton flash kernel (fwd+bwd) in an `nn.Module`/custom op with
  `scaled_dot_product_attention`-compatible signature, including causal masks.
- Benchmark FlexAttention fwd+bwd at 1024/2048 and consider paged/varlen
  variants.
- Keep math SDPA fallback for unsupported shapes/head dims.

### 4. cuBLASLt fallback completeness
- Support `RELU`/`GELU` (and AUX) epilogues; batched GEMM via
  `rocblas_gemm_strided_batched_ex`; row-major orders.
- Add unit tests comparing against cuBLAS-path results.

### 5. Upstreaming
- Submit the `ldmatrix.m8n8.x2.trans.b16` shim (or add the missing function to
  `zluda_ptx_impl.cpp` and regenerate both bitcode blobs with the documented
  LLVM build).
- Submit RDNA1 `dp4a` fallback, gcn_arch feature stripping, and the ld/st
  cache-hint parser relaxations.

### 6. Packaging
- Release zip profiles: default (unified stack), inference (fp16 accumulate),
  GEMM (HIP 6.4 + tuned rocBLAS).
- Launcher: detect tuned rocBLAS libraries and warn if absent.
- Automated benchmark harness with JSON output and regression tracking.

### 7. Explicitly out of scope
- SageAttention (user decision).
- xformers / flash-attn source builds for sm80+ CUTLASS kernels: no RDNA1
  support; would require major porting work and matrix-core features that the
  hardware lacks.
