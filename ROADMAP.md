# Roadmap — ZLUDA on gfx1010 (Radeon RX 5700 XT)

Status: ZLUDA v7-preview.21 builds and runs on gfx1010 with a unified ROCm 10.1
runtime + tuned rocBLAS 7.1.1. See `RESULTS_RDNA1.md` for measurements and
`docs/RDNA1_BUILD_AND_USAGE.md` for build/usage instructions.

## Агентный аудит (2026-11-28)

6 агентов-исследователей + верификатор перепроверили стек по репо,
JSON-результатам, логам и первоисточникам: **77/90 пунктов верифицировано**,
найдено **12 расхождений** доков с фактами и 2 бага обвязки.
Полный отчёт: `docs/AGENT_AUDIT_REPORT.md`.

Исправлено сразу: автозагрузка `cudnn_fallback` (BN/RNN), merge
`TUNED_M1_EXT` в `gemv._config`, replay перед проверкой logits в
`bench/graph_decode.py`, реализован `zluda_ops.auto.uninstall()`;
обновлены устаревшие места в `INFRA_FIXES` (21/25), `KERNEL_GAPS`
(sgemv), `COMPAT_MATRIX` (-fa 1 + software-fallback MMA PR #551),
`ENGINE_COMPARISON` (графы, llama.cpp), `INTEGRATION`, `ROADMAP`
(bf16-emul), `ROADMAP_ECOSYSTEM` (RNN fallback).
Дополнительно сделано: GPTQ/AWQ-распаковка (`unpack_gptq_int4`/`unpack_awq_int4`, selftest 6e-5), GPU-CI (`bench/regression_gate.py` + `tools/run_gpu_ci.ps1`, PASS, baseline 10 метрик), HIP-GEMV harness (`bench/hip_gemv_proto.py`), метка «sdpa (fast off)» в rematrix.
Топ-приоритеты из аудита: P0 стабилизация драйвера/safe-run;
P1 llama.cpp FA/MMQ, torch.compile decode, проводка fused QKV/GQA,
ComfyUI e2e, MIOpen RNN, GPTQ/AWQ, профайлер+GPU-CI.


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
| **P0** | Infra: roofline microbenchmarks, correctness + performance regression tests, clock/power monitoring | **PARTIAL** (protocol done, clocks blocked) | basis for all decisions | low | ADL live status NOT_SUPPORTED |
| **P0** | fp16-accumulation UX: safe opt-in, accuracy table, warnings | **DONE** | ~2x fp16 (inference) | low | HH requires explicit flag |
| **P0** | Tensile tuning: SS / HH_HPA / HH across LLM and SD shapes | **DONE (HH) / CLOSED (SS, HPA)** | HH achieved; SS/HPA on random data are at the practical ceiling — the 8.8–9.2 TF target came from a zero-data artifact (see `bench/rocblas_gap.py`) | medium | rocBLAS 7.1.1 |
| **P0** | GEMV / skinny GEMM (M=1..32) + batched / strided-batched GEMM | **CLOSED** | decode (LLM inference) | medium | M16 176 GB/s (39%), M32 83 (19%); target 70-80% unreachable with split-K/tl.dot/packed-fp16 (all measured worse); rocBLAS fp32 M=32 fallback kept |
| **P0** | Triton FA integration: op/nn.Module, SDPA-compatible backend, GQA/MQA, causal/sliding window | **DONE** | training + inference | medium | masks/dropout/window in Triton; varlen uses per-sequence Triton (fused packed kernel open) |
| **P1** | Flash-decoding (split-KV) for long-context decode at small batch | **DONE** | 3.2–4.5x decode vs math at 1k–4k, rel err <5e-4 | medium | `zluda_ops/flash_decode.py` + auto dispatch |
| **P1** | int4/int8 weight-only **storage** + dequant-to-fp16 GEMV/GEMM; no int8 math on RDNA1 | **PARTIAL** | 4x/2x compression (7B: 3.5/7 GB); decode fused GEMV **1.92x** vs fp16 / 4.6x vs fp16-weight GEMM at M=1; fused prefill GEMM at parity with cached fp16 at M=64, 0.75x at M=128-256 (int4 memory kept); dequant-per-call 1.8-4.6x slower; GPTQ/AWQ exact loading open | high | GEMV/GEMM |
| **P1** | GGUF k-quants (Q4_K/Q5_K/Q6_K) dequant-to-fp16 for llama.cpp HIP; no int8 dot/MMQ acceleration | **OPEN** | LLM on 5700 XT | high | outside ZLUDA |
| **P1** | bf16 emulation: convert once per tile, high-half mask | **DONE** | 4.4→7.92 TF | medium | `zluda_ops.bf16` |
| **P1** | Fused kernels: RMSNorm/LayerNorm (fwd+bwd), RoPE, SwiGLU/GeGLU, softmax, chunked cross-entropy | **DONE** | RMSNorm 8.2x fwd / 6.1x bwd, softmax 18.5x, RoPE 5.1-8.8x, SwiGLU 1.2-1.5x vs torch eager; chunked CE -28% peak memory but slower than torch | medium | Triton |
| **P1** | Winograd/MIOpen conv audit: solver availability, correctness, workspace, speed on 3x3 stride-1/dilation-1 | **DONE** | fp16 anomaly root-caused: MIOpen picks Winograd (`ConvBinWinogradRxSf2x3g1`) only for fp32, fp16 gets GEMM (`GemmFwdRest`). Workaround `zluda_ops/conv.py`: cast-to-fp32 Winograd conv, **1.4-4.3x fwd / up to 1.9x fwd+bwd**, opt-in `ZLUDA_CONV_FP32=1`; channels_last unusable; no gfx1010 perf-db | medium | MIOpen; no ZLUDA-native kernels |
| **P2** | MIOpen solver/perf-db for gfx1010, channels_last+fp16, VAE tiling, GroupNorm+SiLU (SD/ComfyUI) | **PARTIAL** | find-db part **DONE**: user db auto-populated, solver already optimal (FAST/SEARCH change nothing), cold start is 1.2 s process init; channels_last unusable; VAE/GN+SiLU work open | high if SD is a target | medium-high | measured |
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
| 2 — Attention | **PARTIAL** | masks/dropout/window in Triton, varlen per-sequence, flash-decoding done; fused decode/backward integration open |
| 3 — Quantization | **PARTIAL** | groupwise int4/int8 storage + fused int4 dequant-GEMV (decode) and fused dequant-GEMM (prefill) in `zluda_ops/quant.py`; exact GPTQ/AWQ loading open; no int8 compute |
| 4 — Fused kernels | **PARTIAL** | Triton RMSNorm/RoPE/SwiGLU/GeGLU/softmax/AdamW implemented and faster (`zluda_ops/fused.py`); LayerNorm fwd-only win (training slower); chunked CE memory-only; sampling open |
| 5 — Diffusion/MIOpen | **PARTIAL** | conv audited (`bench/conv.py`): NCHW-only, GEMM-fallback, fp16 slower; MIOpen perf-db for gfx1010 missing; channels_last/VAE/auto-selection open |
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
  M=8 242 (fp16), M=16/32 are compute-bound (176/83); rocBLAS linear is
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
- **2026-11-25 - P0-5 monitoring protocol:** `bench/common.py` now records the
  measurement protocol in every JSON report and offers `bench_median` (median
  of per-round means; shuffle variant order between rounds when comparing).
  `bench/gpu_clocks.py` talks to `atiadlxx.dll`: the adapter is enumerated, but
  OverdriveN/6/5 live status returns `ADL_ERR_NOT_SUPPORTED` (-8) on this
  driver, so sclk/mclk/temperature/power remain unavailable; utilization and
  VRAM come from Windows performance counters (`bench/monitor.py`).
- **2026-11-24 - Stage 2.1 (partial):** `zluda_ops/flash_attention.py` exposes
  Triton FA as an autograd op and SDPA-compatible function/module. Causal and
  non-causal fwd+bwd work; GQA works. Masks/dropout/sliding/varlen are
  functional via math fallback but not yet Triton-accelerated. Flash-decoding
  remains open.
- **2026-11-25 - P0-2 (masks):** boolean and additive attention masks (2D, 3D
  per batch, 4D per batch+head, broadcast, optionally with causal) now run in
  the Triton fwd+bwd kernels; broadcast masks stay stride-0, masked lanes are
  zeroed after the exponential (the `-1e6` trick otherwise leaked into long
  rows). `bench/fa_mask.py`: fwd err ≤1e-3, grad err ≤4e-3; masked causal
  s=1024 is 1.22 ms vs 2.04 ms math (1.67x). Dropout/window/varlen remain math
  fallbacks.
- **2026-11-25 - P0-2b (dropout):** inverted dropout runs in the Triton fwd+bwd
  kernels with the Philox mask regenerated from (head, query, key) offsets;
  unbiased (z≈0.8 over 16 seeds), deterministic per seed, p=0 unchanged vs math;
  s=1024 causal p=0.1 is 1.04 vs 1.97 ms (1.90x).
- **2026-11-25 - P1-1 follow-up (fp16 conv):** MIOpen logs prove the dtype
  split — fp32 3x3 uses the Winograd solver `ConvBinWinogradRxSf2x3g1`, fp16
  falls back to `GemmFwdRest` (GEMM). Implemented `zluda_ops/conv.py`
  (cast-to-fp32 Winograd path, gradients flow through the casts) and an opt-in
  `F.conv2d` patch in `zluda_ops.auto` behind `ZLUDA_CONV_FP32=1`. Measured with
  the warmed/shuffled protocol: **1.41-4.25x forward** on 5/5 SD/ResNet shapes,
  **1.02-1.90x forward+backward** on 4/5 (the smallest shape is
  conversion-bound), UNet blocks 1.72x/2.30x, gradient error 2e-8..7e-4.
  `bench/conv_fp16.py` reproduces it.
- **2026-11-25 - P1-5 (fused kernels):** `zluda_ops/fused.py` adds Triton
  RMSNorm (fwd+bwd), RoPE rotate-half (fwd+bwd), SwiGLU (fwd+bwd), row softmax
  and a memory-saving chunked cross-entropy, with `bench/fused.py` comparing
  against torch eager. At 8192x4096 fp16: RMSNorm 8.2x forward and 6.1x
  forward+backward, softmax 18.5x, RoPE (1x32x2048x128) 5.1x/8.8x, SwiGLU
  1.2x/1.5x; errors are fp16 accumulation-level. Chunked CE reduces peak memory
  from 1024 to 736 MB at 4096x16384 but is slower than torch (38 vs 17 ms), so
  it stays a memory-only option. LayerNorm/GeGLU/AdamW/sampling remain open.
- **2026-11-25 - P1-3 (quantization storage):** `zluda_ops/quant.py` implements
  groupwise symmetric int4/int8 weight-only storage (groups 32/64/128, fp16
  scales, two nibbles per byte) plus reference dequantizers and a fused Triton
  int4 dequant-GEMV that never materialises fp16 weights. Compression 4x/2x
  (7B: 3.5/7 GB vs 14 GB); fused GEMV matches the fp32 reference within 2.4e-4
  and is 1.44-1.63x faster than the fp16 GEMV at M=1 (0.078 vs 0.113 ms),
  0.89x at M=8 after fixing register spills (smaller BK, contiguous x pair
  loads). The 70-80% packed-bandwidth target is not met (114-128 GB/s packed):
  dequantisation is compute/launch bound, and a tiled/split-K int4 kernel is
  the next step. GPTQ/AWQ-compatible layout is documented; their exact nibble
  order/zero-point loading and the prefill dequant-GEMM remain open.
- **2026-11-26 - Phase 2 finished (tuning round):** sampling parallelised
  (138.6 us standalone on 151936 vocab vs 658/3693 us before; greedy exact,
  top-50 40/40), GEMV autotuner added (bench/autotune_gemv.py ->
  zluda_ops/tuned_gemv_ext.py, merged as fallback in gemv._config), int8
  prefill tuned (best 387/1624 us at M=64/512 vs fp16 dequant 207/726 us —
  kept as VRAM option only). Final bench/phase2_kernels.py: 9/9 OK. Phase 2
  complete for the planned kernel set; open follow-ups: wire fused QKV/GQA
  kernels into the layer path, more e2e measurements.- **2026-11-26 - Phase 2 COMPLETE (kernels validated + e2e wired):**
  9/9 kernel checks pass (bench/phase2_kernels.py): GQA KV-reuse decode 1.34x,
  M=1 specialised decode 1.34x (M>1 exact after fixing the bench reference),
  fused residual+RMSNorm, fused QKV+RoPE+KV-append, embedding gather+scale,
  Gumbel-max sampling (greedy exact, top-50 40/40, distribution matches
  softmax), int8 prefill GEMM (correct; slower than dequant-once+fp16, kept as
  a VRAM option). Wired fused decoder-layer norm+MLP into
  zluda_ops/model_patches.py: GraphDecoder replay 11.02 -> 10.36 ms/token
  (90.8 -> 96.6 tok/s, +6.4%), logits err 0.034. Roofline: copy 85%, fill 95%,
  large GEMV up to 77% of 448 GB/s peak; small kernels launch-bound. Fixed a
  Triton/ZLUDA codegen bug (constexpr N in add_rms_norm) by using runtime N;
  tl.cumsum is unreliable here so sampling uses Gumbel-max. Remaining:
  int8-prefill tuning, parallel sampling reduction, autotune tables, wire
  fused QKV/GQA into the layer path.- **2026-11-26 - Phase 2 kernels validated (7/7):** GQA KV-reuse decode
  (1.32x vs head-per-program; 87.5 vs 115.9 us), specialised M=1 decode
  (1.26x, err 2e-6), fused residual+RMSNorm (43x: 1.0 vs 46.2 us), fused
  QKV+RoPE+KV-append (err ~1e-3), embedding gather+scale (21.3 us, exact).
  Known issue: general M>1 decode path still disagrees with the math reference
  (err 3.26) - tracked as open item; M=1 decode paths (dominant) are exact.
  Remaining Phase-2 work: sampling kernels, int8/int4 prefill tiles, autotune
  tables + roofline harness, wiring the validated kernels into GraphDecoder.
- **2026-11-26 - Phase 2 started (kernels written, validation pending):**
  implemented `zluda_ops/gqa_decode.py` (GQA KV-reuse flash-decoding: one
  program per (batch, kv-head) split, looping the group heads so K/V hits L2
  instead of DRAM - removes up to 7x redundant KV traffic) and
  `fused.add_rms_norm` (fused residual add + RMSNorm, returns the pre-norm
  residual). Added `bench/phase2_kernels.py` (correctness + speed) and
  docs/PHASE2_KERNELS.md. GPU is wedged from the compatibility run and needs a
  reboot before these can be validated.- **2026-11-26 - compatibility matrix (pre-Phase-2):** added docs/COMPAT_MATRIX.md
  with crash-isolated results. Attention: prefill math/Triton-FA/flex
  (slow)/flash_attn/xformers work; PyTorch sdpa_default/flash/efficient/cudnn
  FAIL (launch failure / cutlass fmha sm80 arch fatal) -> zluda_ops.auto now
  disables the fast SDPA backends; decode: flash-decoding 0.121 ms vs math
  0.305 (2.5x). llama.cpp: F16 fa0 and F32 fa0 work (pp64 1762/2841, tg4
  172/139), fa1 fails (named symbol not found), Q4/Q8 MMQ not runnable safely.
  torch.compile: prefill works with math SDPA (6449-6733 tok/s), fails with raw
  SDPA (cuDNN fmha) or with zluda_ops.auto (Inductor Triton Error); decode
  compile (StaticCache) fails; reduce-overhead unsupported. ComfyUI: base mode
  did not reach API within 200 s under ZLUDA, custom-nodes launcher fails on
  comfyui-sam3 missing comfy_env; 0.5 MP Flux GGUF e2e not completed (needs
  matching text encoders + more VRAM; plan after fresh reboot with minimal
  node set at 704x704).
- **2026-11-26 - PHASE 1 COMPLETE (acceptance suite PASS):** implemented
  `tools/run_infra_suite.ps1` (infra probe + training suite + shim smoke):
  infra 21/25, training 7/7, shims 4/4 -> PASS. Closed in this phase: cuFFT and
  cuSPARSE now work, cublasSgemv works, cuDNN BN/RNN have documented fallbacks,
  flash_attn/xformers decode paths fixed (0.16 ms per attention call), bits-
  andbytes shim over optim8bit, training grads exact (Linear/Conv), Triton FA
  backward, AMP, 8-bit AdamW; int8 fused GEMV and fused SwiGLU GEMV merged;
  hipMalloc==cudaMalloc under ZLUDA. Remaining documented blockers: llama.cpp
  FA/Q4 (ZLUDA PTX), ComfyUI e2e (Flux GGUF needs text encoders/VRAM and a fresh
  boot run), torch.compile+StaticCache (Inductor codegen), fp64 (no Tensile),
  CUPTI profiling. Phase 2/3 start from here. Details:
  docs/ROADMAP_ECOSYSTEM.md.
- **2026-11-26 - fused SwiGLU GEMV merged + model patch:** the earlier NaN in
  the fused gate/up+SwiGLU kernel was a missing K-bound mask (out-of-bounds tile
  reads), not a Triton codegen bug; with the mask it is correct (err 0.0078) and
  faster than the 3-kernel path (70.9 vs 98.3 us at 4864x896 = 1.39x, 32.5 vs
  70.5 us at 896x896 = 2.17x). zluda_ops/model_patches.py patches the Qwen2 MLP
  decode path to use it; end-to-end graph-replay decode gain ~1%
  (10.83 -> 10.74 ms/token at 1k) as the kernel saving is partly hidden by the
  rest of the layer. Removed leftover debug eprintln from ZLUDA stream/graph
  capture paths. torch.compile + StaticCache remains blocked.
- **2026-11-26 - torch.compile + StaticCache blocked:** compiling the
  StaticCache decode path fails in Inductor kernel generation
  (`Triton Error [CUDA]: error`); prefill compile (plain model, DynamicCache)
  still works (+42-49%). Decode delivery stays CUDA-graph replay
  (`zluda_ops.graph_decode.GraphDecoder`). Open item: identify the failing
  inductor kernel.
- **2026-11-26 - cublasSgemv fixed:** zluda_blas had no level-2 BLAS at all, so
  torch.mv on fp32 failed with CUBLAS_STATUS_NOT_SUPPORTED. Added sgemv /
  sgemv_v2 / sgemv_v2_64 mapping to rocBLAS (+vtable/implemented wiring);
  torch.mv fp32 and fp16 now work. cublasDgemv (fp64) remains unsupported (no
  FromCuda conversion for *const f64; fp64 BLAS absent on this gfx1010 build).
  Infra probe: 21/25 OK.
- **2026-11-26 - kernel gaps + allocation audit:** added fused int8 GEMV
  (`quant.int8_linear`, correct 2e-4..4e-3) - fused is memory-optimal but
  dequant-once+tuned fp16 GEMV is faster for compute (45.6 vs 61.9 us qkv,
  69.5 vs 123.1 gate/up). Attempted fused gate/up+SwiGLU GEMV: potential ~2x
  but the Triton kernels produced wrong/NaN output on this stack (codegen issue,
  left as open item). Missing-kernel list documented (GQA KV-reuse decode,
  fused QKV+RoPE+KV append, fused residual+norm, sampling, embedding gather).
  New library gaps found: cublasSgemv (1-D fp32 GEMV) NOT_SUPPORTED, fp64 GEMM
  unsupported. hipMalloc vs cudaMalloc on memory-bound kernels: identical
  (fill 421 vs 426 GB/s, copy 384 vs 383, lm_head GEMV 358 GB/s both; only
  allocation latency differs 0.254 vs 0.212 ms). Details: docs/KERNEL_GAPS.md.
- **2026-11-26 - cuDNN BN/RNN fallbacks:** cuDNN batchnorm and RNN are not
  mapped in zluda_dnn8; `zluda_ops/cudnn_fallback.py` provides two fallbacks:
  BatchNorm forwards run with cuDNN disabled (native path, auto-loaded by
  zluda_ops.auto, `ZLUDA_CUDNN_FALLBACK=0` to disable) and
  `rnn_context()` for LSTM/GRU (creation+forward inside the context, because the
  backend is chosen before the call; retrying after a failed cuDNN call leaves
  MIOpen state unusable). Infra probe: 20/24 OK (added fallback checks);
  remaining: raw BN/RNN calls (use fallbacks), fp64 GEMM (no rocBLAS/Tensile
  f64 for gfx1010).
- **2026-11-26 - Phase-1 infra fixes: cuFFT + cuSPARSE now work.** cuFFT:
  PyTorch's FFT path uses `cufftXtMakePlanMany`/`cufftXtExec`, which were
  unimplemented (CUFFT_NOT_SUPPORTED); implemented them mapping cudaDataType
  pairs to hipFFT transform types and dispatching hipfftExec* through a
  per-plan kind registry. torch.fft/ifft and rfft2/irfft2 round-trips are exact.
  cuSPARSE: implemented SetStream/GetStream (rocsparse_set_stream, plus the
  missing hip_runtime-sys dep), Xcoo2csr (rocsparse_coo2csr) and CreateCsr
  (rocsparse_create_csr_descr with datatype mapping); torch.sparse COO mm is
  exact (err 0.0). Full infra probe: 18/22 OK (was 16/22). Remaining:
  cuDNN batchnorm/RNN (work with cudnn.enabled=False native fallback) and fp64
  GEMM (cublasDgemm NOT_SUPPORTED). Details: docs/INFRA_FIXES.md,
  docs/ROADMAP_ECOSYSTEM.md; probe: bench/infra_probe.py.
- **2026-11-25 - ecosystem roadmap (3 phases) + Phase-1 analysis:**
  docs/ROADMAP_ECOSYSTEM.md created: Phase 1 close the infrastructure (status
  matrix of every library/engine/application with evidence, verification
  checklist, ranked fixes), Phase 2 custom fast Triton kernels for everything
  (kernel inventory + roofline targets), Phase 3 replace hipBLAS/rocBLAS hot
  paths with custom kernels (only where phase 2 leaves a gap). Phase-1 probes
  already found: cuFFT FAIL (CUFFT_NOT_SUPPORTED for torch.fft), cuSPARSE FAIL
  (cusparseSetStream not supported), cuDNN batchnorm FAIL
  (CUDNN_STATUS_NOT_SUPPORTED), RNN/LSTM FAIL (CUDNN_STATUS_INTERNAL_ERROR +
  MIOpen kernel argument error, process died); pooling/convT/randn/topk/sort/
  bmm/sdpa work. Top fixes ranked: cuSPARSE stream, cuFFT plans, cuDNN
  batchnorm, RNN/LSTM, then llama.cpp FA/Q4, torch.compile decode, shims, ComfyUI.
- **2026-11-25 - Triton optimality + custom HIP kernels audit:** Triton GEMV
  reaches 15-40% of bandwidth on per-layer decode matvecs (68-181 GB/s of
  448 GB/s) and flash-decoding wastes 7x KV bandwidth on GQA; FA prefill (33%
  of fp16 peak) and fused kernels are near practical limits. A custom HIP
  warp-per-row fp16 GEMV prototype (clang 20 -> gfx1010 DLL, launched from
  Python via ctypes) is 1.1-4.8x faster than Triton (qkv 4.9 vs 23.5 us, gate
  27.6 vs 48.2, down 29.9 vs 49.7, lm_head 751 vs 820 us; 292-362 GB/s), which
  would cut ~3.3 ms/token (~30%) from the graph-decode step. Integration caveat:
  raw HIP launches on stream 0 are not ordered with ZLUDA streams (races, hang)
  and per-call device sync costs ~140 us; CUDA-graph capture of external HIP
  launches is unverified. Recommendation: do not replace the stack; focus on
  Triton GQA KV-reuse + fused QKV/gate-up + multi-row GEMV, and only build a HIP
  GEMV library once ZLUDA exposes proper stream/graph integration. Custom GEMM
  is not worth it (rocBLAS+Tensile at 84% of fp32 peak). Docs:
  docs/TRITON_HIP_AUDIT.md; prototype: tools/hip_proto/.
- **2026-11-25 - int4 solver audit:** decode-shape comparison of the fused int4
  GEMV vs dequant-once + tuned fp16 GEMV (M=1, group 64): dequant-once wins on
  every shape (43 vs 63 us qkv, 70 vs 120 gate/up, 75 vs 87 down, 832 vs 2329 us
  lm_head, i.e. up to 2.8x), while fused int4 stays the memory-optimal choice
  (4x smaller weights). For prefill the fused tiled int4 GEMM remains faster
  than dequant-once. Recommendation documented in docs/SOLVER_AUDIT.md.
- **2026-11-25 - auto conv gate fixed:** auto.py installs the F.conv2d patch for
  ZLUDA_CONV_AUTO=1 as well (line-based edit after CRLF mismatch).
- **2026-11-25 - conv solver selection:** implemented per-shape solver
  auto-selection in `zluda_ops/conv.py` (fp32 winograd weights cached by
  data_ptr/_version, one-time timing of fp16 vs fp32 candidates per shape,
  decision cached; `ZLUDA_CONV_AUTO=1`, legacy `ZLUDA_CONV_FP32=1` still works).
  Measured: small shapes stay on fp16 (0.10 ms), mid/large shapes take the
  cached-weight fp32 Winograd path - 1x128x32x32 0.12 vs 0.16 ms, 1x320x16x16
  0.15 vs 0.22 ms, 1x64x128x128 0.22 vs 0.58 ms (2.7x); error 2e-3..4e-3.
  Remaining audit items: int4 decode-shape configs, fused int8, rocFFT/SPARSE
  paths (not exercised).
- **2026-11-25 - solver/kernel audit (first pass):** GEMV configs were a single
  fixed (4096,4) - swept and replaced with a shape-aware TUNED_M1 table
  (qkv 28.5 us 2.07x vs rocBLAS, gate/up 40.1 us 2.01x, down_proj 53.7 us
  5.29x, lm_head 798 us 12.2x). FA prefill config swept at 512..8192: current
  config already optimal (<=1.05x). Flash-decode SPLITS heuristic already
  optimal (1.00x) and 2.5-6.8x faster than math SDPA, but GQA was unsupported -
  added native GQA indexing and dispatch (correctness 3e-5); graph decode now
  89.7 tok/s @1k / 72.3 @2k. Remaining: conv solver selection, int4 decode
  shapes, fused int8. Details: docs/SOLVER_AUDIT.md.
- **2026-11-25 - reusable graph decoder:** added `zluda_ops/graph_decode.py`
  (`GraphDecoder`: StaticCache prefill + CApture of one decode step + replay,
  token/position buffers updated outside the graph). Replay-only decode:
  **8.01 ms/tok (124.8 tok/s) at 256 ctx**, 87.6 tok/s at 1k, 72.8 at 2k; full
  generation with token readback 119 tok/s. Eager PyTorch 25-27 tok/s, llama.cpp
  Vulkan ~238 tok/s. lm_head matvec 151936x896: rocBLAS 9.73 ms vs Triton GEMV
  0.86 ms (11x). The self-loop graph variant (token update inside the graph) is
  not usable on ZLUDA (~1 tok/s, repeated tokens). docs/CUDA_GRAPHS_ZLUDA.md.
- **2026-11-25 - bf16 emulation works on ZLUDA:** with in-place parameter
  conversion (bf16 storage -> fp32 compute) the ZLUDA process runs bf16-emul at
  **5488 tok/s prefill / 29.6 tok/s decode / 2073 MB**, beating both native ZLUDA
  bf16 (3450/11.1) and ROCm bf16-emul (3812/26.0). The earlier "crash" was a
  harness invocation without the ZLUDA launcher, not a GPU/VRAM issue.
- **2026-11-25 - graph-replayed decode works (zLUDA):** found and fixed the
  decode GEMV dispatch bug in `zluda_ops.auto` (decode F.linear gets (1,1,D)
  tensors, so the 2-D-only Triton GEMV path never triggered - 0 calls, rocBLAS
  silently used). With the flatten fix a decode step makes ~170 Triton GEMV
  calls and no cuBLAS, which makes the whole step graph-capture safe:
  torch.cuda.CUDAGraph capture/replay now works. Measured (Qwen2.5-0.5B fp16,
  StaticCache, bench/graph_decode.py): eager 25.1/26.7 tok/s vs **graph-replay
  87.6 tok/s @1k and 72.8 tok/s @2k** (3.5x / 2.7x). llama.cpp on Vulkan is
  ~238 tok/s @2k, so PyTorch/ZLUDA is now within 2-3x instead of 8x. Remaining:
  tune Triton GEMV for the exact decode shapes (lm_head 151936x896), support
  position updates with the graph in a real generation loop, flash-decoding
  tuning. Details: docs/CUDA_GRAPHS_ZLUDA.md.
- **2026-11-25 - decode/CUDA-graphs follow-up:** measured the decode gap
  components: launches cost only ~2 us (240 launches ~0.5 ms/token), while
  rocBLAS GEMV costs 24.7-251 us per matvec (down_proj 1x896x4864 = 251 us vs
  Triton 35.8 us, 7x faster); PyTorch eager decode 34.0 ms/token vs llama.cpp
  Vulkan 4.2 ms/token. So the Vulkan advantage is kernel efficiency
  (bandwidth-bound hand-written GEMV + few submissions per token), not launch
  overhead; CUDA graphs alone cannot close it. torch.compile reduce-overhead now
  reaches graph instantiation and fails on (a) DynamicCache torch.cat inside
  the captured region (use StaticCache) and (b) cuBLAS not being graph-capture
  safe (zluda_blas audit is the next item). bf16-emul works on ROCm (3812/26.0)
  and on ZLUDA too (5488/29.6 tok/s); the earlier "crash" was a harness
  invocation without the ZLUDA launcher, not a VRAM limit.
- **2026-11-25 - CUDA graphs work under ZLUDA:** torch.cuda.CUDAGraph capture
  and replay now work. PyTorch's runtime path called three unimplemented legacy
  driver entry points - `cuStreamBeginCapture` (no mode arg),
  `cuStreamGetCaptureInfo` (3-arg) and `cuGraphInstantiate` (no flags) - which
  made `torch.cuda.graph()` fail with "operation not supported" at
  capture_begin. Added them (stream.rs/graph.rs + implemented list), rebuilt
  nvcuda.dll; verified driver-level (ctypes capture/instantiate/launch, bit
  exact) and PyTorch-level (capture/replay, max err 0.0). This is the key
  enabler for the decode gap (launch-bound: 28-51 tok/s PyTorch vs 184-238 tok/s
  llama.cpp Vulkan). Debug logging for unimplemented APIs is opt-in via
  ZLUDA_DEBUG_UNIMPLEMENTED=1. Docs: docs/CUDA_GRAPHS_ZLUDA.md; harness:
  bench/graph_decode.py. Next: torch.compile reduce-overhead retest, graph
  decode benchmark, bf16-emul retry.
- **2026-11-25 - cross-backend comparison (Vulkan vs native ROCm vs ZLUDA):**
  built F32/BF16 GGUF variants for Vulkan/llama.cpp; found the native ROCm torch
  environment in `D:\comfyui\Data\Packages\ComfyUIrocm\venv` (torch
  2.14.0+rocm10.1.0, HIP 7.16) and ran the math-SDPA matrix there. ZLUDA beats
  native ROCm torch on this model (fp16 4831 vs 3082 prefill / 28.8 vs 19.8
  decode; fp32 5418 vs 1177; bf16 3450 vs 2845). fp16 accumulation is the
  biggest PyTorch knob on ZLUDA (7450 prefill math, 9460 with Triton FA) and is
  a cuBLAS concept (ignored by ROCm, absent on Vulkan). ZLUDA attention stacks
  measured: math, SDPA-fast (falls back to math), Triton FA (+16% prefill),
  flex attention (works), flash_attn/flex shims (prefill good, decode needs
  work), xformers shim OOM. Vulkan llama.cpp GGUF matrix
  F16/F32/BF16 measured (high variance across runs). Full report:
  docs/CROSS_BACKEND_COMPARISON.md; raw data in
  bench/results/cross_backend_summary.json.
- **2026-11-25 - llama.cpp under ZLUDA WORKS (custom PTX build), same-engine
  Vulkan comparison:** built llama.cpp locally with portable CUDA 12.4.1
  (nvcc 12.4.131, VS 2022) and `-DCMAKE_CUDA_ARCHITECTURES=86-virtual`
  (PTX only - the official SASS-only builds have no code for sm_86 and cannot
  run on ZLUDA). Result: `-ngl 99` full GPU offload, stable driver, F16 GGUF
  works; FA=1 (mma.sync) and Q4_K_M (MMQ) still fail/hang on ZLUDA. Same-engine
  measurement (Qwen2.5-0.5B F16, fa=0): ZLUDA prefill 6144/3294/732 tok/s and
  decode 184.2/184.6/183.9 tok/s at 2k/8k/32k vs Vulkan (Ollama) 6258/4179 and
  212.3/172.7 tok/s at 2k/8k (F16 clamped to 8192 on Vulkan). ZLUDA is within
  -2% prefill at 2k, -21% at 8k, decode -13%/+7% at 2k/8k, and can run 32k
  context where Vulkan clamps. Full report:
  docs/LLAMACPP_VULKAN_VS_ZLUDA.md; recipe: docs/LLAMACPP_ZLUDA_BUILD.md;
  raw data: bench/results/llamacpp_zluda_vs_vulkan.json.
- **2026-11-25 - ecosystem shims:** added `integrations/flash_attn` and
  `integrations/xformers` shims (flash_attn_func/varlen_func/bert_padding,
  xformers.ops.memory_efficient_attention + bias types) backed by the Triton FA
  and flash-decoding kernels with a math-SDPA fallback, installable with
  `tools/install_ecosystem.py` (venv `.pth`). CPU API/fallback tests pass
  (flash_attn err 4e-7, xformers ops finite); GPU smoke tests of the shims are
  the next safe step. llama.cpp remains blocked (driver hard-crash with
  offloaded layers) and is documented in INTEGRATION/ENGINE_COMPARISON.
- **2026-11-25 - llama.cpp verdict:** after the launch-attribute patch ZLUDA can
  run llama.cpp only at `-ngl 0` (CPU: pp16 124/tg2 17.4) or `-ngl 1 -fa 0`
  (F16: pp8 415.6/tg1 61.5). Scaling GPU offload fails with launch errors or
  hard-crashes the AMD driver (reboot required); Q4_K_M hangs at ngl=1. This is
  a ZLUDA runtime/compiler limitation (llama.cpp kernel set + PTX JIT), not a
  llama.cpp binary bug, so a same-engine Vulkan-vs-ZLUDA comparison stays
  blocked. The launch-attribute patch is kept as a general improvement; the
  matrix runner is ready for when ZLUDA can execute llama.cpp kernels. Next
  ecosystem work continues with safe paths (torch.compile/SDPA/ComfyUI and API
  shims for flash_attn/xformers backed by the Triton kernels).
- **2026-11-25 - llama.cpp under ZLUDA (patch attempt):** ZLUDA's
  `launch_kernel_ex` rejected PDL launch attributes; patched
  `zluda/src/impl/driver.rs` to accept launch-time attributes (serialized
  execution is semantically correct) and rebuilt `nvcuda.dll`. llama.cpp (b12721,
  CUDA 12.4) then loads under ZLUDA and sees the GPU; `-ngl 0` works through the
  CUDA backend (pp16 124 tok/s, tg2 17.4), but `-ngl >= 1` hangs with every
  workaround (graphs off, no pinned memory, no fusion, force-MMQ, PDL off) and
  the pre-PDL build b9802 errors in `ggml-cuda.cu`. Blocker is now ZLUDA's
  execution/JIT of llama.cpp's CUDA kernels (async-copy pipeline / cp.async),
  not the launch API - requires extending the ptx/llvm_zluda support. Same-engine
  Vulkan-vs-ZLUDA comparison stays blocked; the cross-engine table in
  ENGINE_COMPARISON.md is labelled accordingly.
- **2026-11-25 - engine comparison, final pass:** Vulkan (Ollama, 100% GPU)
  Q4_K_M: 4531/2741/946 tok/s prefill and 150.8/135.2/104.8 tok/s decode at
  2k/8k/32k; F16 clamped to 8192 context by Ollama on 8 GB. ZLUDA/PyTorch fp16:
  4525 tok/s prefill at 1k (on par), **7774 with fp16 accumulation** (+70%),
  6449-6733 with torch.compile (peak 1031 MB); long context 3275/2076/1199 tok/s
  prefill at 8k/16k/32k and 12.6/6.7/6.2 tok/s decode. **Decode is 4-8x (17x at
  32k) slower than Vulkan** - the main remaining gap (per-token launch
  overhead, no CUDA graphs, math attention fallback; our flash-decoding gives
  +22%). torch.compile is unusable for decode here (recompiles per token;
  CUDA-graph capture unsupported by ZLUDA). Quantization: steady fp16 950 MB ->
  int4 519 MB (1.85x) but prefill slower; int8 dequant-once cache doubles
  active memory -> fused int8 kernel needed. llama.cpp CUDA under ZLUDA is
  blocked in both current (cudaLaunchKernelEx/PDL) and pre-PDL builds;
  bitsandbytes is excluded (substitute Triton 8-bit AdamW: 4.6x faster step,
  3.94x smaller state). Explicit RMSNorm/SwiGLU fusion patch gives only 1.01x.
  Full report with the "what else can be implemented" analysis:
  ENGINE_COMPARISON.md.
- **2026-11-25 - engine comparison (Vulkan vs ZLUDA), first pass:** Ollama
  Vulkan runs Qwen2.5-0.5B 100% on GPU (Q4_K_M: 4531 tok/s prefill and 150.8
  tok/s decode at 2k context, 946/104.8 at 32k; F16 is clamped to 8192 context).
  ZLUDA/PyTorch: prefill 4525 tok/s fp16, **7774 tok/s with fp16 accumulation**
  (faster than Vulkan), 4984 fp32, 2798 bf16; decode 19-28 tok/s, i.e. **4-8x
  slower than Vulkan** due to per-token launch overhead and the math attention
  fallback - the main ZLUDA-side gap. llama.cpp-CUDA under ZLUDA is blocked by
  `cudaLaunchKernelEx`/PDL (hangs with PDL disabled); bitsandbytes is excluded
  (`named symbol not found`), replaced by a Triton blockwise-int8 AdamW
  (4.6x faster step, 3.94x smaller state). Fixed a recursion bug in the Triton
  FA math fallback. Full data: ENGINE_COMPARISON.md.
- **2026-11-25 - P1-3 prefill extension:** added `quant.int4_matmul` (fused
  dequant-GEMM, weights never materialised as fp16) and
  `quant.int4_prefill_linear` (dequant-once reference). At N=K=4096 the fused
  GEMM is 1.8-4.6x faster than dequantize-per-call and 1.00-0.75x of cached
  fp16 weights (parity at M=64, 25-30% slower at M=128-256) while keeping the
  4x memory saving; decode remains 4.6x faster than fp16 weights. Tiled int4
  GEMV (+6%, inside clock variance) is not adopted. `bench/quant_prefill.py`
  reproduces the table.
- **2026-11-25 - fused kernels extension:** added exact-erf GeGLU fwd+bwd,
  LayerNorm fwd+bwd and an in-place fused AdamW to `zluda_ops/fused.py`. Clean
  measurements: RMSNorm 9.4x fwd / 2.95x fwd+bwd, softmax 2.3x, RoPE 10.9x/8.3x,
  SwiGLU 1.5x, GeGLU 1.54x/1.49x, AdamW **3.4x** (param diff 3e-8), LayerNorm
  forward 1.19x but training 0.53x (torch native LayerNorm wins because of the
  atomic gradients), chunked CE memory -28% but slower. GPU benchmarks were
  re-run in a verified idle window after a concurrent external job wedged the
  driver; results in RESULTS_RDNA1.md.
- **2026-11-25 - MIOpen find-db/solver:** no system find-db for gfx1010 is
  shipped, but the user find-db (`~/.miopen`) is populated automatically and used
  (~4.5 ms per new shape). The ~1.2 s first-convolution cost is process
  initialisation, not find. `MIOPEN_FIND_MODE=FAST` and
  `MIOPEN_FIND_ENFORCE=SEARCH` do not change steady-state throughput (SEARCH
  only adds 45-124 ms to the first call), so the selected solvers are already
  optimal for the tested shapes. `bench/miopen_db.py` reproduces this. The
  remaining conv opportunity is the dtype solver split, addressed by
  `zluda_ops/conv.py`.
- **2026-11-25 - P1-2 (flash-decoding):** `zluda_ops/flash_decode.py`
  splits the KV cache into up to 16 chunks per (batch, head), computes partial
  (acc, m, l) and merges with the standard log-sum-exp formula. Decode shapes
  (B1 H8 D64 fp16, M=1/4/8) match the float32 reference within 4.6e-4 relative;
  speed vs math SDPA: 3.39x/3.33x/3.55x at N=1k/2k/4k for M=1, 4.50x for
  M=8 causal at 4k, 1.86x at 2k. `zluda_ops.auto` now routes
  `F.scaled_dot_product_attention` decode calls (M≤16, KV≥512, no grad/mask/
  dropout) to flash-decoding (1.33x through the Python patch, 3.35x raw kernel).
- **2026-11-25 - P1-1 (conv audit):** `bench/conv.py` measures ResNet/SD 3x3
  convolutions. MIOpen logs prove the selected solver is `GemmFwdRest`
  (`miopenConvolutionFwdAlgoGEMM`, find 0.1052 ms): 3x3 is lowered to a rocBLAS
  GEMM on gfx1010, Winograd/implicit-GEMM are not selected. NCHW is the only
  usable layout (channels_last 10-1000x slower); fp16 conv is 2-4x slower than
  fp32; the gfx1010 perf-db (`gfx1010_20.HIP.fdb.txt`) is missing and the CK
  grouped-conv library is unavailable. Decision: no custom Winograd path; the
  remaining win is the fp16-conv anomaly and shipping a perf-db.
- **2026-11-25 - P0-4 (GEMV M=16/32):** benchmarked a split-K / tl.dot kernel
  (`bench/gemv_m32.py`) and packed-fp16 product variants. Split-K reaches
  64-81 GB/s at M=32 (mvec 81), packed-fp16 is worse (31-50), so the 70-80%
  bandwidth target is recorded as unreachable on this reduction pattern. Best
  remaining win: M=16 with fp32 products, **176 GB/s (39%, +13%)**, now the
  default benchmark config; M=32 stays at 83 GB/s (19%), where fp32 rocBLAS
  (119 GB/s) remains the auto-path fallback. Peak GEMV M=1 improved to
  302 GB/s / 331 GB/s (fp16/fp32).
- **2026-11-25 - P0-3 (window/varlen):** sliding window is implemented
  arithmetically in the Triton fwd+bwd kernels (block-range skipping, no dense
  mask; `lo` aligned down to `BLOCK_N` before advancing block pointers).
  `flash_attn_varlen_func` dispatches per sequence through Triton with no dense
  masks. `bench/fa_window.py`: fwd/grad err ≤2e-3 (window) and ≤4e-3 (varlen
  window); s=1024 window (128,0) is 0.26 ms vs 1.87 ms math (7.1x). A fused
  packed varlen kernel remains open.
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
| GEMV M16-32 target | **CLOSED** | 176/83 GB/s (39%/19%); 70-80% target not reachable; split-K/tl.dot/packed-fp16 measured worse; rocBLAS fp32 M=32 stays the fallback |
| FA causal/non-causal/GQA | **DONE** | Triton fwd+bwd |
| FA masks | **DONE** | bool/additive 2D/3D/4D broadcast masks, causal+mask, fwd+bwd in Triton; fwd err ≤1e-3, grad err ≤4e-3; masked causal 1024 is 1.67x math |
| FA dropout | **DONE** | inverted dropout in Triton fwd+bwd, unbiased (z<1), deterministic per seed, fwd+bwd errors unchanged; causal 1024 1.90x math |
| FA window | **DONE** | arithmetic sliding window in fwd+bwd, block-range skipping, no dense mask; fwd/grad err ≤2e-3; s=1024 window(128,0) 0.26 ms vs 1.87 ms math (7.1x) |
| FA varlen | **DONE (per-sequence)** | packed q/k/v + `cu_seqlens` dispatch, Triton per sequence without dense masks (window/grad err ≤4e-3); a fused packed varlen kernel remains open |
| Flash-decoding | **DONE** | split-KV + LSE merge; 3.35x (M=1) / 4.50x (M=8 causal) decode at 4k vs math, rel err ≤4.6e-4; auto-dispatched from SDPA |
| bf16 native / emulation | **DONE (emulation)** | 4.42 native / 7.92 emulated TF |
| GPU monitoring | **PARTIAL** | utilization/VRAM via Windows PDH; measurement protocol enforced in every report; ADL live clocks return NOT_SUPPORTED (-8), documented blocker |
| int8 compute | **IMPOSSIBLE** | storage/dequant only |
| Winograd / MIOpen conv | **AUDIT DONE, TEST OPEN** | solvers found; benchmark/auto-select pending |

## Next actions (ordered)

| # | Action | Priority | Gate/acceptance |
|---|---|---|---|
| 1 | Winograd/MIOpen conv test (`bench/conv.py`) | P1 | **CLOSED 2026-11-25**: MIOpen selects `GemmFwdRest` (GEMM) for 3x3; Winograd not selected; NCHW-only; fp16 2-4x slower than fp32; perf-db for gfx1010 missing — decision: no custom Winograd path, follow-ups = fp16 conv + perf-db |
| 2 | Triton dropout for FA | P0 | **CLOSED 2026-11-25**: masks and inverted dropout run in Triton fwd+bwd; window/varlen remain (row 3) |
| 3 | Triton sliding-window + varlen FA | P0 | **CLOSED 2026-11-25**: window arithmetic in kernel (7.1x @1024), varlen per-sequence Triton; fused packed varlen kernel still open |
| 4 | GEMV M=16/32 tuning | P0 | **CLOSED 2026-11-25**: M16 176 GB/s (+13% via fp32-product config), M32 83; split-K/tl.dot (64-81) and packed-fp16 (31-50) measured worse; target 70-80% recorded as unreachable, rocBLAS fp32 M=32 kept as fallback |
| 5 | SS/HPA Tensile | P0 | **CLOSED 2026-11-25**: random-data ceiling 8.16/6.94 TF; 325/302 solutions already enumerated, default is best, no toolchain for new kernels |
| 6 | Clock/power monitoring | P0 | **DONE 2026-11-25**: protocol (random data, shuffled rounds, median) enforced in `bench/common.save_report`; utilization/VRAM via Windows PDH; sclk/mclk/temp are a documented blocker (`bench/gpu_clocks.py`: ADL adapters enumerate but OverdriveN/6/5 return ADL_ERR_NOT_SUPPORTED = -8) |
| 7 | Flash-decoding split-KV | P1 | **CLOSED 2026-11-25**: 3.35x (M=1) and 4.50x (M=8 causal) at 4k vs math, rel err ≤4.6e-4; auto-dispatches SDPA decode via `zluda_ops.auto` |
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
