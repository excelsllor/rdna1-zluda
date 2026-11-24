# ZLUDA for RDNA1 (gfx1010) — Radeon RX 5700 XT

Fork of [vosen/ZLUDA](https://github.com/vosen/ZLUDA) focused on AMD RDNA1
(gfx1010). Windows build with a ROCm 10.1 runtime and tuned gfx1010 rocBLAS;
runs PyTorch (CUDA 11.8 wheels), Triton, `torch.compile`, and Triton flash
attention with working backward.

- **Before / after:** [`BEFORE_AFTER.md`](BEFORE_AFTER.md)
- **Out-of-the-box acceleration:** [`INTEGRATION.md`](INTEGRATION.md)
- **Results:** [`RESULTS_RDNA1.md`](RESULTS_RDNA1.md)
- **Roadmap:** [`ROADMAP.md`](ROADMAP.md)
- **Build and usage guide:** [`docs/RDNA1_BUILD_AND_USAGE.md`](docs/RDNA1_BUILD_AND_USAGE.md)
- **Patch against upstream v7-preview.21:** [`local-patches/zluda-rdna1-gfx1010.patch`](local-patches/zluda-rdna1-gfx1010.patch)

## At a glance (RX 5700 XT, 4096² GEMM)

| Path | FP32 | FP16 (fp32 accum) | BF16 |
|---|---|---|---|
| ROCm 10.1 stock rocBLAS (fallback Tensile) | 4.85 TF | 4.21 TF | 2.32 TF |
| **ROCm 10.1 + tuned rocBLAS 7.1.1 (supported)** | **8.13–8.18 TF** | **6.93–6.96 TF** | **4.42 TF** |

Only the unified ROCm 10.1 + tuned rocBLAS 7.1.1 stack is supported by this fork.

- FP16 with fp16 accumulation (inference opt-in): **10.4–12.7 TF**.
- Triton flash attention (B1H8D64, causal): fwd 0.48 / 1.08 / 3.09 ms at
  s=512 / 1024 / 2048 (math SDPA: 0.65 / 1.79 / 7.66 ms); forward+backward
  1.42 / 3.70 / 10.92 ms, gradients match math SDPA.
- FlexAttention (compiled): 0.50 ms at s=512.
- AMP training (small transformer block): 65.3 ms/step on ZLUDA vs 131.8 ms/step
  on native ROCm 10.1.

## What this fork changes

1. `gfx1010` support in `llvm_zluda`: strip `:xnack-` target features from
   `gcn_arch` (ISA parse + LLVM CPU name); disable DIA/PDB when ATL is absent.
2. `dp4a` software fallback for targets without `v_dot4` (RDNA1).
3. PTX parser accepts and ignores `ld`/`st` cache-policy / eviction-priority /
   cache-hint / prefetch qualifiers (required by Triton and TorchInductor).
4. `llvm_zluda` IR shim for `__zluda_ptx_impl_ldmatrix_m8n8_x2_trans_b16`
   (missing from the shipped bitcode; needed by Triton attention backward).
5. `zluda_blaslt`: cuBLASLt→rocBLAS fallback with descriptor tracking, per-row
   bias epilogue, and fp16-compute descriptors (hipBLASLt has no gfx1010
   kernels).
6. Diagnostics for module compilation and `isa_version` in the cache key.

## Branches

- `main` — snapshot of the working tree (no upstream CI files).
- `rdna1-gfx1010` — full upstream history plus the changes above.

## Quick start

```bat
git clone --recursive https://github.com/excelsllor/rdna1-zluda.git
cd rdna1-zluda
build-env.cmd cargo xtask zip --release
```

Runtime: use the unified ROCm 10.1 user space (HIP 7.16, MIOpen) with
rocBLAS 7.1.1 + tuned gfx1010 Tensile libraries. Python environment:
`torch==2.7.1+cu118` + `triton-windows==3.3.1`. See
[`docs/RDNA1_BUILD_AND_USAGE.md`](docs/RDNA1_BUILD_AND_USAGE.md) for the exact
steps, environment variables (`HIP_PATH`, `ZLUDA_CC=8.6`,
`TRITON_PTXAS_PATH`), launcher (`run-zluda.cmd`),
ComfyUI notes, and troubleshooting.

```bat
run-zluda.cmd venv\Scripts\python.exe train.py
run-zluda.cmd venv\Scripts\python.exe bench\matmul_bench.py
run-zluda-auto.cmd venv\Scripts\python.exe inference.py
run-zluda-train.cmd path\to\train.py
```

## Known limitations

- PyTorch SDPA `flash` and `memory_efficient` are not available on gfx1010
  (kernels exist only for MFMA architectures); use math SDPA or Triton flash.
- No tensor cores/MFMA and no `v_dot4`: FP16 peaks require packed half2 math
  with fp16 accumulation; dp4a is emulated.
- hipBLASLt has no gfx1010 kernels; cuBLASLt calls go through the rocBLAS
  fallback (bias is supported, other epilogues are limited).
- RDNA1 fixed 8.6 compute capability is required for Triton (`ZLUDA_CC=8.6`).
- KV int8/fp8 is not on the roadmap. `xformers`/`flash-attn` CUDA compatibility
  is tracked as a timeboxed P2 workstream.

## Upstream and license

Based on ZLUDA v7-preview.21 (commit `eb89ae3`) by Andrzej Janik and
contributors. See the upstream project for CUDA compatibility details and
documentation: <https://github.com/vosen/ZLUDA>. Licensed under Apache-2.0 or
MIT, matching upstream.
