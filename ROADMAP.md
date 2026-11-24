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
- ZLUDA compiles device code with `-wavefrontsize64,+cumode` (wave32 + CU mode).
- Known limitations: PyTorch SDPA flash/memory-efficient kernels are not
  available on gfx1010 (no MFMA); `cuXXX` performance libraries are mapped to
  ROCm equivalents where possible (hipBLASLt has no gfx1010 kernels, so a
  rocBLAS fallback is used).

## Priority matrix

| Priority | Workstream | Benefit | Effort | Dependencies |
|---|---|---|---|---|
| **P0** | Infra: roofline microbenchmarks, correctness + performance regression tests, clock/power monitoring | basis for all decisions | low | — |
| **P0** | fp16-accumulation UX: safe opt-in, accuracy table, warnings | ~2x fp16 (inference) | low | done, needs UX |
| **P0** | Tensile tuning: SS / HH_HPA / HH across LLM and SD shapes | +8–15% GEMM, HH 12→14–16 TF | medium | rocBLAS 7.1.1 |
| **P0** | GEMV / skinny GEMM (M=1..32) + batched / strided-batched GEMM | decode (LLM inference) | medium | benchmarks |
| **P0** | Triton FA integration: op/nn.Module, SDPA-compatible backend, GQA/MQA, causal/sliding window | training + inference | medium | fwd+bwd exists |
| **P1** | Flash-decoding (split-KV) for long-context decode at small batch | 1.5–2x decode | medium | FA integration |
| **P1** | int4/int8 weight-only **storage** + dequant-to-fp16 GEMV/GEMM; no int8 math on RDNA1 | largest memory win for 8 GB | high | GEMV/GEMM |
| **P1** | GGUF k-quants (Q4_K/Q5_K/Q6_K) dequant-to-fp16 for llama.cpp HIP; no int8 dot/MMQ acceleration | LLM on 5700 XT | high | outside ZLUDA |
| **P1** | bf16 emulation: convert once per tile, high-half mask | 4.4→6–7 TF | medium | Tensile/LLVM |
| **P1** | Fused kernels: RMSNorm/LayerNorm (fwd+bwd), RoPE, SwiGLU/GeGLU, softmax, chunked cross-entropy | memory/speed | medium | Triton |
| **P2** | MIOpen solver/perf-db for gfx1010, channels_last+fp16, VAE tiling, GroupNorm+SiLU (SD/ComfyUI) | high if SD is a target | medium-high | measure first |
| **P2** | GEMM epilogues (bias/activation/residual/scale), grouped GEMM (MoE) | +5–15% | medium | Tensile |
| **P2** | Triton/Inductor autotune configs for RDNA1; WGP vs CU experiment | +5–20% on kernels | low-medium | benchmarks |
| **P2** | xformers / flash-attn CUDA compatibility: source builds for cu118, PTX execution through ZLUDA, API-compatible wrapper + benchmarks | ecosystem compatibility | high | cu118 toolchain; risk: sm80 kernels and no matrix cores |
| **P2** | Allocator (`PYTORCH_HIP_ALLOC_CONF`), HIP Graphs, offload, gfx1010 runtime stability | OOM/latency/stability | medium | tests |
| **P3** | Paged attention / vLLM, prefix caching, speculative decoding | serving only | high | decide on serving |
| **P3** | 8-bit optimizer storage only (no int8 compute); bitsandbytes/SmoothQuant int8 kernels rejected | memory only | high | — |
| **P3** | Distribution (wheels/Docker), CI on GPU, ExLlamaV2/A1111/onnx integrations | usability | medium | — |
| **Icebox** | SageAttention (user decision); int8 GEMM/tensor acceleration (hardware absent); ROCm rollback; Vulkan/OpenCL backend | rejected/out of scope | — | — |

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
- Solver/perf-db work (implicit GEMM, Winograd, direct fp16), channels_last,
  VAE tiling, GroupNorm+SiLU, SD attention with small heads.
- Only if ComfyUI/SD/Flux is a primary target.

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
  Open: re-tuned Tensile for fp32 8.8-9.2/HH 14-16, M=16/32 GEMV,
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
- **2026-11-24 - Stage 2.1 (partial):** `zluda_ops/flash_attention.py` exposes
  the causal Triton FA as an autograd op, SDPA-compatible function and module;
  GQA works, fwd 1.6-2.4x faster than math SDPA, grads within 0.006.
  Non-causal backward and sliding/varlen/softcap/dropout fall back or remain
  open.

## Definition of done (metrics)
- fp32 GEMM 8.18→**8.8–9.2 TF**; HPA 6.96→**7.5–8**; HH 12.3→**14–16**;
  bf16 4.42→**6–7**.
- int4 weight-only decode (dequant-to-fp16): ≥**70–80%** of 448 GB/s;
  7B int4 decode target **35–60 tok/s** (theoretical ~100).
- No int8 GEMM/TOPS target: gfx1010 has no int8 dot/tensor units; int8 is
  storage/dequantization only.
- Flash fwd 2048: 3.09→**≤2.8 ms**; flash-decoding 4k+: **≥1.5–2x**;
  gradient error ≤**1e-3**.
- Chunked CE: logits memory reduced by the chunk factor.
- MIOpen/SD: measurable conv/UNet speedup in ComfyUI over the current unified
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
