# Roadmap — ZLUDA on gfx1010 (Radeon RX 5700 XT)

Status: ZLUDA v7-preview.21 builds and runs on gfx1010 with a unified ROCm 10.1
runtime + tuned rocBLAS 7.1.1. See `RESULTS_RDNA1.md` for measurements and
`docs/RDNA1_BUILD_AND_USAGE.md` for build/usage instructions.

## Facts and constraints

- gfx1010: 40 CU / 20 WGP, wave32, **no matrix units (no tensor cores/MFMA)**,
  **no int8 dot (`v_dot4`) and no int8 tensor units**. int8 is storage-only or
  must be dequantized/emulated to fp16; there is no int8 GEMM acceleration.
  Packed FP16 (`v_pk_fma_f16`) only with fp16 accumulation.
- Peaks: FP32 9.75 TFLOPS, FP16 19.51 TFLOPS (packed 2:1 only), no bf16
  hardware. Memory 448 GB/s, 8 GB VRAM, PCIe 4.0 x16 (~25 GB/s host link).
- Runtime: ROCm 10.1 user space (HIP 7.16, MIOpen) + rocBLAS 7.1.1 with tuned
  gfx1010 Tensile libraries. PyTorch `2.7.1+cu118` (PTX path), Triton 3.3.1.
- Benchmarks must use random data and median timing: zeros, ones and
  uninitialised buffers raise the measured GEMM rate by ~12% (fp32 4096:
  8.19 TF random vs 9.23 TF zeros) through power/clock effects, not through a
  software path. Verified by `bench/rocblas_gap.py`; the old 9.2–9.6 TF
  readings came from uninitialised `hipMalloc` buffers.
- ZLUDA compiles device code with `-wavefrontsize64,+cumode` (wave32 + CU mode).
- Known limitations: PyTorch SDPA flash/memory-efficient kernels are not
  available on gfx1010 (no MFMA); `cuXXX` performance libraries are mapped to
  ROCm equivalents where possible (hipBLASLt has no gfx1010 kernels, so a
  rocBLAS fallback is used).

## Priority matrix

| Priority | Workstream | Status | Benefit | Effort | Dependencies |
|---|---|---|---|---|---|
| **P0** | Infra: roofline microbenchmarks, correctness + performance regression tests, clock/power monitoring | **PARTIAL** | basis for all decisions | low | clocks/power blocked |
| **P0** | fp16-accumulation UX: safe opt-in, accuracy table, warnings | **DONE** | ~2x fp16 (inference) | low | HH requires explicit flag |
| **P0** | Tensile tuning: SS / HH_HPA / HH across LLM and SD shapes | **DONE (HH) / CLOSED (SS, HPA)** | HH achieved; SS/HPA on random data are at the practical ceiling — the 8.8–9.2 TF target came from a zero-data artifact (see `bench/rocblas_gap.py`) | medium | rocBLAS 7.1.1 |
| **P0** | GEMV / skinny GEMM (M=1..32) + batched / strided-batched GEMM | **PARTIAL** | decode (LLM inference) | medium | M=16/32 target open |
| **P0** | Triton FA integration: op/nn.Module, SDPA-compatible backend, GQA/MQA, causal/sliding window | **PARTIAL** | training + inference | medium | masks/dropout/window/varlen are math fallback |
| **P1** | Flash-decoding (split-KV) for long-context decode at small batch | **OPEN** | 1.5–2x decode | medium | FA integration |
| **P1** | int4/int8 weight-only **storage** + dequant-to-fp16 GEMV/GEMM; no int8 math on RDNA1 | **OPEN** | largest memory win for 8 GB | high | GEMV/GEMM |
| **P1** | GGUF k-quants (Q4_K/Q5_K/Q6_K) dequant-to-fp16 for llama.cpp HIP; no int8 dot/MMQ acceleration | **OPEN** | LLM on 5700 XT | high | outside ZLUDA |
| **P1** | bf16 emulation: convert once per tile, high-half mask | **DONE** | 4.4→7.92 TF | medium | `zluda_ops.bf16` |
| **P1** | Fused kernels: RMSNorm/LayerNorm (fwd+bwd), RoPE, SwiGLU/GeGLU, softmax, chunked cross-entropy | **OPEN** | memory/speed | medium | Triton |
| **P1** | Winograd/MIOpen conv audit: solver availability, correctness, workspace, speed on 3x3 stride-1/dilation-1 | **AUDIT DONE / TEST OPEN** | SD/CNN speedup if MIOpen wins | medium | MIOpen; no ZLUDA-native kernels |
| **P2** | MIOpen solver/perf-db for gfx1010, channels_last+fp16, VAE tiling, GroupNorm+SiLU (SD/ComfyUI) | **OPEN** | high if SD is a target | medium-high | measure first |
| **P2** | GEMM epilogues (bias/activation/residual/scale), grouped GEMM (MoE) | **PARTIAL** | +5–15% | medium | measured, not fused/auto |
| **P2** | Triton/Inductor autotune configs for RDNA1; WGP vs CU experiment | **OPEN** | +5–20% on kernels | low-medium | benchmarks |
| **P2** | xformers / flash-attn CUDA compatibility: source builds for cu118, PTX through ZLUDA, API wrapper + benchmarks | **OPEN** | ecosystem compatibility | high | sm80 kernels; no matrix cores |
| **P2** | Allocator (`PYTORCH_HIP_ALLOC_CONF`), HIP Graphs, offload, gfx1010 runtime stability | **PARTIAL** | OOM/latency/stability | medium | ComfyUI workaround; no full fix |
| **P3** | Paged attention / vLLM, prefix caching, speculative decoding | **OPEN** | serving only | high | decide on serving |
| **P3** | 8-bit optimizer storage only; bitsandbytes/SmoothQuant int8 kernels rejected | **OPEN** | memory only | high | no int8 units |
| **P3** | Distribution (wheels/Docker), CI on GPU, ExLlamaV2/A1111/onnx integrations | **OPEN** | usability | medium | no GPU CI yet |
| **Icebox** | SageAttention; int8 GEMM/tensor acceleration (hardware absent); ROCm rollback; Vulkan/OpenCL backend | **REJECTED** | out of scope | — | — |

Status keys: **DONE**, **PARTIAL**, **OPEN**, **BLOCKED**, **REJECTED**, and **AUDIT DONE / TEST OPEN**.

## Stage status

| Stage | Status | Note |
|---|---|---|
| 0 — Foundation | **PARTIAL** | roofline/regression done; clocks/power blocked |
| 1 — GEMM | **PARTIAL** | baseline, bf16, HH done; SS/HPA, M16/32, fused epilogues open |
| 2 — Attention | **PARTIAL** | causal/non-causal/GQA done; masks/dropout/window/varlen fallback; flash-decoding open |
| 3 — Quantization | **OPEN** | storage path not implemented; no int8 compute |
| 4 — Fused kernels | **OPEN** | not started |
| 5 — Diffusion/MIOpen | **PARTIAL** | Winograd audit done; benchmark and auto-selection open |
| 6 — Runtime/memory | **PARTIAL** | ComfyUI workaround; allocator/graphs/offload open |
| 7 — Compiler/LLVM/Triton | **OPEN** | WGP/CU and autotune not started |
| 8 — Integrations/distribution | **PARTIAL** | ComfyUI auto/GGUF works; wheels/CI/other apps open |
| Parallel — Upstreaming | **PARTIAL** | gfx1010 fixes in fork; upstream PRs not done |

## Stages

### Stage 0 — Foundation (1–2 days)
- Roofline microbenchmarks (ALU, memory bandwidth, LDS, latencies).
- Correctness regression tests with dtype tolerances; performance grid
  (GEMM/attention/GEMV).
- Clock/power monitoring (sclk, mclk, temperature) and a fixed profile for
  reproducible results.

### Stage 1 — GEMM (≈1 week)
1. Tensile 7.1.1 tuning for SS / HH_HPA / HH over LLM shapes (K 1–16k,
   M/N 1–8k, batched) and SD shapes; ship `.dat`/`.co` sets and a launcher
   profile; regression tests.
2. GEMV/skinny GEMM (M=1..32): Triton kernel with dwordx4 loads, unpack and
   conversion, fp16 activations; target 70–80% of 448 GB/s.
3. Batched/strided-batched GEMM; bias/activation/residual epilogues.
4. bf16: convert once per tile into LDS/registers, mask for the high half;
   target 6–7 TF.
5. int8 compute is not applicable on gfx1010 (no `v_dot4`/int8 units).
   Evaluate int4/int8 **storage** + dequant-to-fp16 for bandwidth savings only;
   do not schedule an int8 GEMM path.

### Stage 2 — Attention (≈1 week)
1. Package Triton FA (fwd+bwd) as an op/nn.Module with an
   `scaled_dot_product_attention`-compatible API, GQA/MQA, causal, sliding
   window, varlen, softcap/dropout.
2. Flash-decoding (split-KV + log-sum-exp merge) for decode at small batch.
3. Benchmarks 512–8k against math SDPA; acceptance: ≥1.5x fwd at 2k+, decode
   ≥1.5x at 4k+.
4. Prefix caching / speculative decoding at framework level after Stage 3.

### Stage 3 — Quantization (2–3 weeks; storage/bandwidth only)
1. int4/int8 weight-only **storage**: dequant-to-fp16 GEMV (decode) and
   dequant+GEMM (prefill), GPTQ/AWQ group-size 32/64/128; accuracy validation
   vs fp16/fp32. No int8 accumulation or int8 tensor path exists on RDNA1.
2. GGUF k-quants for the llama.cpp HIP backend: dequant-to-fp16 (no dp4a/int8
   dot, no MMQ acceleration), tracked outside ZLUDA.
3. 8-bit optimizers are storage-only; bitsandbytes/SmoothQuant int8 kernels
   are rejected because the hardware has no int8 units.

### Stage 4 — Fused kernels (≈1 week)
- RMSNorm/LayerNorm fwd+bwd, RoPE, SwiGLU/GeGLU, softmax, chunked
  cross-entropy, fused AdamW/clip, GPU sampling, gather/scatter/index_select.
- Write only where Inductor is worse; inspect with `TORCH_LOGS=output_code`
  first.

### Stage 5 — Diffusion / MIOpen (1–2 weeks, after measurements)
- Measure conv/batchnorm/RNN on gfx1010 in the unified stack.
- Winograd/MIOpen test: verify solver availability and whether it beats implicit GEMM; see the dedicated section below.
- Solver/perf-db work (implicit GEMM, Winograd, direct fp16), channels_last,
  VAE tiling, GroupNorm+SiLU, SD attention with small heads.
- Only if ComfyUI/SD/Flux is a primary target.

### Winograd / convolution test (P1, timeboxed)

**Audit result:** ZLUDA has no native Winograd kernels. `zluda_dnn` maps
`miopenConvolutionAlgoWinograd` to the cuDNN Winograd algorithm and MIOpen.dll
contains Winograd solvers/kernels (`ConvMPBidirectWinograd`,
`ConvWinograd3x3MultipassWrW`, `miopenGcnAsmWinogradXform*`). Availability and
speed on gfx1010 are **not yet measured**.

**Test matrix:**
- Shapes: 3x3 stride 1 / dilation 1, fp16 and fp32, batch 1/8, channels
  64/128/256/512; plus representative ResNet/SD UNet shapes.
- Layouts: NCHW first; channels_last only if MIOpen advertises it.
- Compare Winograd vs implicit GEMM vs direct; correctness vs CPU; workspace;
  forward and backward.
- Instrument with `MIOPEN_LOG_LEVEL=6`/`MIOPEN_FIND_MODE`, record the chosen
  solver, add `bench/conv.py`, and write results to `RESULTS_RDNA1.md`.

**Acceptance:** enable/auto-select Winograd only if it is >=10% faster than the
best non-Winograd solver at equal accuracy and fits the workspace. Otherwise
record "available but not beneficial" and do not add a custom path.

**Risks:** MIOpen may reject gfx1010 for some shapes; cuDNN mapping does not
guarantee selection; transforms add memory traffic and may lose accuracy.

### Stage 6 — Runtime and memory (parallel)
- Allocator tuning (`expandable_segments`, `max_split_size_mb`,
  `garbage_collection_threshold` — verify support).
- HIP Graphs for decode; offload of weights/KV/optimizer with overlap;
  gradient checkpointing; streaming safetensors loading.
- Fix gfx1010-specific crashes/page faults and unsupported architecture checks.

### Stage 7 — Compiler / LLVM / Triton (parallel)
- WGP vs CU experiment (drop `+cumode`), `s_clause`/`s_waitcnt`/VMEM grouping,
  `-O3`/fast-math where valid.
- Autotune configs for wave32 and Inductor tables; correctness gates after
  every change.

### Stage 8 — Integrations and distribution
- xformers / flash-attn CUDA compatibility (build from source for cu118,
  execute sm80 PTX through ZLUDA, API-compatible wrappers; benchmark and
  document limits).
- llama.cpp HIP backend, ExLlamaV2/V3, ComfyUI/A1111/Forge, onnxruntime.
- Wheels/Docker for easy installation; CI with local GPU runs.

### Parallel track — Upstreaming
- `ldmatrix.m8n8.x2.trans.b16` shim (or regenerate the bitcode), RDNA1 dp4a
  fallback, PTX cache-hint parsing, cuBLASLt→rocBLAS fallback; tests included.

## Progress log

- **2026-11-24 — Stage 0 (partial):** added `bench/roofline.py` and
  `bench/regression.py`. 12/12 correctness checks pass; 4096 GEMM 8.16 TF
  fp32 / 6.91 fp16 / 4.40 bf16. Roofline: FP32 ALU 5.0 TF (51%, Triton
  codegen), FP16 ALU 18.8 TF (96%), memory 373–378 GB/s (83–85%), HBM latency
  172 ns, cache-resident load ~3.1 TB/s. Clock/power monitoring is blocked:
  ZLUDA NVML returns NotSupported and the ROCm 10.1 user space has no
  rocm-smi/amd-smi on Windows.
- **2026-11-24 - Stage 1 (baseline complete):** `bench/gemv.py` adds a Triton
  GEMV/skinny kernel for M=1..32. M=1 294/311 GB/s fp16/fp32 (66/69%), M=4 278,
  M=8 242 (fp16), M=16/32 are compute-bound (155/78); rocBLAS linear is
  108-155 GB/s and `cublasSgemv` fp32 is unsupported. `bench/stage1.py`
  records the GEMM shape grid (fp32 8.16, fp16 6.92, bf16 4.40 TF at 4096),
  batched/strided GEMM (7.5/6.9 TF), bias/residual/GELU epilogues, bf16 fp32
  emulation (7.92 TF at 4096, meets the 6-7 TF target). int8 compute is absent
  on gfx1010 (no `v_dot4`/int8 units), so only storage+dequant remains.
  Open: SS/HPA Tensile (HH done via override), M=16/32 GEMV,
  fused non-bias epilogues, storage-only quantization (Stage 3).
- **2026-11-24 - P0 training stability (fixed for eager):** the
  `bench_zluda_model_compile.py` crash was caused by PyTorch selecting
  unsupported cuDNN/CUTLASS FMHA kernels; the failed kernel attempts then
  surfaced as `cublasSetWorkspace` `<unknown>` in backward. Added
  `zluda_ops.mha` slow-path MHA and `run-zluda-train.cmd` (VRAM 85%, no
  inference auto-patch). Eager training now runs: 1040–1208 ms/step,
  loss 0.00683, 0 FATAL, no `cublasSetWorkspace`. `torch.compile` is skipped
  for this model: it takes ~30 minutes even on an NVIDIA T4, so it is not a
  P0 blocker; use eager training or compile a smaller model.
- **2026-11-24 - P0-5 HH kernels already existed:** the tuned rocBLAS 7.1.1 library contains 110 HH solutions for gfx1010. `tensile_overrides/gfx1010_hh.csv` selects better existing kernels via `ROCBLAS_TENSILE_GEMM_OVERRIDE_PATH`; `run-zluda.cmd` applies it. HH 2048 **17.94 TF**, HH 4096 **16.73 TF** (defaults 9.5/12.1). No Tensile rebuild needed. SS/HPA remain ~8.2/7.2 TF; zeros/ones/uninitialized data inflated earlier numbers and are not representative.
- **2026-11-24 - Stage 2.1 (partial):** `zluda_ops/flash_attention.py` exposes
  Triton FA as an autograd op and SDPA-compatible function/module. Causal and
  non-causal fwd+bwd work; GQA works. Masks/dropout/sliding/varlen are
  functional via math fallback but not yet Triton-accelerated. Flash-decoding
  remains open.
- **2026-11-25 - repository maintenance:** portable launchers (repository-
  relative paths, `vswhere`, `ZLUDA_HIP_PATH`), documented ComfyUI launchers,
  `gpu_reset.ps1`, ComfyUI freeze file, patch export/validation tooling
  (`tools/export_patch.ps1`, `tools/check_patch.ps1`, pre-push hook), regenerated
  UTF-8/LF patch, `smoke/local/` for machine-specific scripts.
- **2026-11-25 - P0-1 closed (fp32 GEMM "gap"):** `bench/rocblas_gap.py` runs
  `torch.mm`, direct `rocblas_sgemm` on torch/hipMalloc buffers and a zeroed
  control in one warmed process with shuffled rounds. Random data: all paths
  8.18–8.19 TF (median). Zeroed buffers: **9.23 TF (+12.8%)** — an artifact
  caused by lower power draw/higher clocks. The old 9.2–9.6 TF readings came
  from uninitialised `hipMalloc` buffers in raw-HIP scripts; there is no
  ZLUDA/torch gap, and the 8.8–9.2 TF SS target is withdrawn.

## P0 status (2026-11-24)

| Item | Status | Note |
|---|---|---|
| Training stability / MHA crash | **DONE (eager)** | slow-path MHA; full compile skipped |
| fp16 accumulation UX | **DONE** | opt-in context, warnings, accuracy table; default off |
| Tensile HH | **DONE** | existing 110 kernels; 16.73/17.94 TF with override |
| Tensile SS/HPA | **CLOSED** | 8.16/6.94 TF on random data = practical ceiling (84% of peak); old 9.2/7.2 TF numbers were zero-data artifacts; 325/302 solutions enumerated, default already best |
| GEMV M1-8 / auto M<=32 | **DONE** | ~2x rocBLAS; inference only |
| GEMV M16-32 target | **PARTIAL** | 155/78 GB/s; target 70-80% not met |
| FA causal/non-causal/GQA | **DONE** | Triton fwd+bwd |
| FA masks/dropout/window/varlen | **FUNCTIONAL FALLBACK** | correct math; Triton path open |
| Flash-decoding | **OPEN** | |
| bf16 native / emulation | **DONE (emulation)** | 4.42 native / 7.92 emulated TF |
| GPU monitoring | **PARTIAL** | utilization/VRAM; clocks/power blocked |
| int8 compute | **IMPOSSIBLE** | storage/dequant only |
| Winograd / MIOpen conv | **AUDIT DONE, TEST OPEN** | solvers found; benchmark/auto-select pending |

## Next actions (ordered)

| # | Action | Priority | Gate/acceptance |
|---|---|---|---|
| 1 | Winograd/MIOpen conv test (`bench/conv.py`) | P1 | >=10% vs implicit GEMM or document "not beneficial" |
| 2 | Triton masks/dropout for FA | P0 | match math fallback within 1e-3, faster at 1k+ |
| 3 | Triton sliding-window + varlen FA | P0 | correctness vs math; no dense mask allocation |
| 4 | GEMV M=16/32 tuning | P0 | >=70% of 448 GB/s |
| 5 | SS/HPA Tensile | P0 | **CLOSED 2026-11-25**: random-data ceiling 8.16/6.94 TF; 325/302 solutions already enumerated, default is best, no toolchain for new kernels |
| 6 | Clock/power monitoring | P0 | sclk/mclk/temp available or documented blocked |
| 7 | Flash-decoding split-KV | P1 | >=1.5x decode at 4k |
| 8 | Full 8-layer `torch.compile` | Deferred | only if compile time becomes acceptable |

## Definition of done (metrics)
- fp32 GEMM 8.16 TF = **84% of theoretical peak**, on random data this is the
  practical ceiling for the shipped Tensile kernels; HPA 6.94 TF = bounded by
  the fp32 rate (`v_fma_mix`); HH target met via override (**16.73/17.94 TF**);
  bf16 native 4.42→6–7 open (needs new Tensile code objects), bf16 emulation
  **7.92 TF** reaches its target.
- int4 weight-only decode (dequant-to-fp16): ≥**70–80%** of 448 GB/s;
  7B int4 decode target **35–60 tok/s** (theoretical ~100).
- No int8 GEMM/TOPS target: gfx1010 has no int8 dot/tensor units; int8 is
  storage/dequantization only.
- Flash fwd 2048: 3.09→**≤2.8 ms**; flash-decoding 4k+: **≥1.5–2x**;
  gradient error ≤**1e-3**.
- Chunked CE: logits memory reduced by the chunk factor.
- MIOpen/Winograd: >=10% faster than the best non-Winograd 3x3 solver on supported shapes, or documented "not beneficial".
  stack.
- Reproducible results: JSON reports, fixed clocks, local/CI runs.

## Risks
- Tensile library format must match the `rocblas.dll` version (regression
  tests required).
- fp16 accumulation changes numerics: opt-in only, with an accuracy table.
- MIOpen may lack the needed solvers/perf-db: measure before investing.
- xformers/flash-attn CUDA kernels target sm80+ SASS/PTX; RDNA1 has no matrix
  cores, so gains may be limited or negative — timeboxed investigation.
- int8 quantization will not produce int8-speed GEMM on RDNA1; only smaller
  storage and dequantization bandwidth savings are realistic.
- Paged/vLLM only if serving is required.

## Rejected approaches
- SageAttention (user decision).
- int8 GEMM/tensor-core acceleration: gfx1010 has no int8 dot (`v_dot4`) or
  int8 matrix units. Only storage + dequant to fp16 is meaningful.
- Rolling back ROCm (breaks ZLUDA/native torch on gfx1010).
- Vulkan/OpenCL backend (loses required functionality).
