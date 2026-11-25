# Engine comparison: Vulkan (Ollama) vs ZLUDA (PyTorch/Triton) on gfx1010

Final report: 2026-11-25. GPU: RX 5700 XT (gfx1010, 8 GB). Model: Qwen2.5-0.5B
(same weights: GGUF for Ollama, safetensors for PyTorch). VRAM capped to 85%
in PyTorch runs. GPU clocks cannot be fixed (+-15-30% run-to-run spread).

## Setups

- **Vulkan**: Ollama 0.40.18, llama.cpp Vulkan backend, 100% GPU offload, GGUF
  Q4_K_M and F16 imported with num_ctx 32768. FA and KV cache type switched via
  `bench/ollama_config.ps1`; measured with `bench/ollama_bench.py` and
  `bench/run_vulkan_matrix.py`.
- **ZLUDA**: this fork (PyTorch 2.7.1+cu118, Triton 3.3.1, ROCm 10.1 + tuned
  rocBLAS 7.1.1), measured with `bench/model_bench.py` (which also reports the
  flash-attention/flash-decoding dispatch counters from `zluda_ops.auto`).
- **llama.cpp CUDA under ZLUDA: patched, still blocked at kernel level.**
  ZLUDA used to reject `cudaLaunchKernelEx` when the PDL launch attribute was
  set (`ERROR_NOT_SUPPORTED`); this fork now accepts launch-time attributes and
  runs the kernels serialized (`zluda/src/impl/driver.rs`; PDL is an overlap
  hint, so serialization is correct) and `nvcuda.dll` was rebuilt.
  With the patch llama.cpp b12721 loads under ZLUDA and sees the GPU
  (8176 MiB, CC 8.6), and `llama-bench -ngl 0` works through the CUDA backend on
  the CPU (pp16 124 tok/s, tg2 17.4 tok/s). But **`-ngl >= 1` hangs** with every
  workaround tried (`GGML_CUDA_DISABLE_GRAPHS=1`, `GGML_CUDA_NO_PINNED=1`,
  `GGML_CUDA_DISABLE_FUSION=1`, `GGML_CUDA_FORCE_MMQ=1`, `GGML_CUDA_PDL=0`), and
  the pre-PDL build b9802 errors in `ggml-cuda.cu`. The remaining blocker is
  ZLUDA's execution/JIT of llama.cpp's CUDA kernels (the async-copy pipeline;
  `cp.async` strings are present in ggml-cuda.dll, while the Triton/PyTorch
  kernels ZLUDA already runs do not use it). Fixing this requires extending
  ZLUDA's `ptx`/`llvm_zluda` support; no llama.cpp binary patch can avoid those
  kernels. Same-engine Vulkan-vs-ZLUDA comparison therefore remains blocked and
  the cross-engine table below is labelled accordingly.
- **bitsandbytes: excluded.** 0.50.2 imports and ships ROCm libraries, but its
  kernels fail under ZLUDA (`named symbol not found`, default and
  `BNB_BACKEND=rocm`). Substitute in this repo: `zluda_ops/optim8bit.py`.

## Vulkan results (Ollama)

Q4_K_M / F16, KV q8_0, FA=1:

| model | ctx | prompt tok | prefill tok/s | decode tok/s |
|---|---|---|---|---|
| Q4_K_M | 2048 | 1850 | 4531 | 150.8 |
| Q4_K_M | 8192 | 7310 | 2741 | 135.2 |
| Q4_K_M | 32768 | 29150 | 946 | 104.8 |
| F16 | 2048 | 1850 | 4771 | 133.7 |
| F16 | 8192 | 7310 | 2810 | 123.3 |
| F16 | 32768 | - | - | - (Ollama clamps F16 to 8192 ctx on 8 GB) |

FA on/off, KV f16 (q8_0 KV requires FA, otherwise HTTP 500):

| model | FA | ctx | prefill tok/s | decode tok/s |
|---|---|---|---|---|
| Q4_K_M | 1 | 8192 | 2769 | 143.1 |
| F16 | 1 | 8192 | 2857 | 133.4 |
| Q4_K_M | 0 | 2048 | 2782 | 196.2 |
| Q4_K_M | 0 | 8192 | 3986 | 178.6 |
| F16 | 0 | 2048 | 6258 | 212.3 |
| F16 | 0 | 8192 | 4179 | 172.7 |

## ZLUDA/PyTorch results (Qwen2.5-0.5B)

Prefill/decode at 1024 exact tokens (decode 8 tokens):

| config | prefill tok/s | decode tok/s | peak VRAM |
|---|---|---|---|
| fp16, math SDPA | 4525 | 22.1 | 1270 MB |
| fp16, fp16-accum | **7774** | 26.2 | 1270 MB |
| fp16, Triton FA + flash-decoding | 4813 | **27.9** | 1281 MB |
| bf16, math SDPA | 2798 | 10.2 | 1270 MB |
| fp32, math SDPA | 4984 | 25.3 | 2518 MB |
| int4 weights (fused dequant) | 2031 | 25.4 | steady 519 MB |
| int8 weights (dequant-once) | 1990 | 18.9 | steady 747 MB, 1694 MB active |

Long context (fp16, Triton FA, exact seq, `logits_to_keep=1`):

| ctx | prefill tok/s | decode tok/s | peak VRAM |
|---|---|---|---|
| 1024 | 4813 | 27.9 | 1281 MB |
| 8192 | 3275 | 12.6 | 1340 MB |
| 16384 | 2076 | 6.7 | 1723 MB |
| 32768 | 1199 | 6.2 | 2487 MB |

torch.compile (fp16, ctx 1024, `mode` in the table, dynamic=False):

| mode | prefill tok/s | decode tok/s | peak VRAM |
|---|---|---|---|
| eager | 4525 | 22.1 | 1270 MB |
| default | **6449** | 0.1 | 1031 MB |
| max-autotune-no-cudagraphs | **6733** | 0.0 | 1031 MB |
| reduce-overhead (CUDA graphs) | unsupported: `CUDA error: operation not supported` on graph capture | | |
| default + dynamic=True | 3612 | 0.27 | 1031 MB |

Explicit fusion test: replacing Qwen2 RMSNorm and SwiGLU MLP with the Triton
kernels from `zluda_ops.fused` gave **1.01x** only in eager mode (Inductor's
whole-graph fusion wins); the fused kernels are useful where `torch.compile`
is not available (ComfyUI, dynamic shapes) and for isolated ops.

## Analysis: where ZLUDA wins / loses

- **Prefill**: ZLUDA is competitive and can win. fp16 4525 tok/s is on par with
  Vulkan (4364-4905); with fp16 accumulation 7774 tok/s (+70% over Vulkan);
  torch.compile raises it to 6449-6733 tok/s. ZLUDA at 32k prefill (1199 tok/s)
  is faster than Vulkan Q4_K_M at 32k (946 tok/s).
- **Decode is the main gap**: 22-28 tok/s at 1k context and 6.2 tok/s at 32k vs
  105-212 tok/s on Vulkan, i.e. 4-8x at short context and ~17x at 32k. Causes:
  one Python/kernel-launch-heavy forward per token, math SDPA for the query in
  the default path (our flash-decoding helps ~+22%), no CUDA/HIP graphs,
  no persistent/fused decode kernels, no batched decode.
- **Torch.compile**: excellent for prefill on this stack (Inductor fuses the
  whole graph, peak memory drops to 1031 MB) but unusable for decode with
  `dynamic=False` (recompilation per token) and CUDA graphs are not supported
  by ZLUDA; `dynamic=True` also recompiles per token here. Decode-specific
  compilation/graph support is an open ZLUDA issue.
- **Quantization**: steady memory is a real win (fp16 950 MB -> int4 519 MB,
  1.85x), but prefill gets slower (2031 tok/s) because dequant ALU is no
  longer hidden at large M; int8's dequant-once cache makes active memory
  worse (1694 MB) - a fused int8 kernel is the clear next step.
- **8-bit AdamW** (bnb substitute): step 0.124 ms vs 0.566 ms torch (4.6x) and
  3.94x smaller optimizer state; 11% parameter deviation after 8 steps due to
  int8 state quantization (same tradeoff as bitsandbytes).

## What else can be implemented (not touched yet)

ZLUDA core / runtime:
- CUDA graph capture/instantiate support (unblocks
  `torch.compile(mode="reduce-overhead")` and would attack the decode gap
  directly); broader CUDA 12 launch API coverage is partly done (launch-time
  attributes/PDL are now accepted, see the llama.cpp section), and the next
  ZLUDA compiler item is support for llama.cpp's async-copy kernel pipeline
  (a prerequisite for the same-engine Vulkan-vs-ZLUDA comparison).
- Native HIP-process ZLUDA (no CUDA shim) and a Vulkan/DX12 backend are out of
  scope but would remove the CUDA-API translation overhead.

Decode path:
- Persistent decode kernel (all 24 layers in one launch), fused QKV + RoPE +
  KV-cache append, fused MLP, CUDA/HIP graph capture of the decode loop,
  batched/defragmented KV cache, paged attention, speculative decoding, and
  KV-cache quantization (int8/fp8 storage) - the decode gap is the largest
  remaining opportunity on this card.

Quantization:
- Fused int8 GEMV/GEMM (analogous to the int4 kernels) so int8 storage does not
  need the 950 MB dequantized copy; tiled int4 GEMM; fused int4 prefill with
  better tiling/vectorization; exact GPTQ/AWQ loading (nibble order, zero
  points, act-order) to drop the 9-12% RTN error; KV-cache quantization;
  int4 embeddings/lm_head; CPU/disk offload of int4 weights.

Kernels / compiler:
- Triton autotune tables for RDNA1 (wave32, WGP vs CU; `+cumode` experiment),
  fused residual+RMSNorm, RoPE+attention, GeGLU/AdamW vectorization,
  chunked-CE backward, flash-decoding at long KV (split tuning), bigger
  flash-attention tile search, MIOpen find-db for gfx1010 (channel_last/VAE
  paths), GEMM epilogue fusion (activation/residual/scale), grouped GEMM for
  MoE, int8/fp8 stubs for libraries that probe them.

Engines / integrations:
- llama.cpp CUDA under ZLUDA (blocked by launch API; see above),
  vLLM/paged-attention port, ExLlamaV2/AWQ, xformers/flash-attn PTX
  compatibility, ComfyUI/SD/Flux real-workload measurements (conv fp16 ->
  fp32 Winograd workaround is available), llama.cpp HIP/ROCm build for a
  native-ROCm baseline on this machine (ROCm 6.4 has no gfx1010 support, so a
  TheRock-based build is needed), ONNX Runtime/TensorRT-like paths are out of
  scope.

Tooling / methodology:
- Fixed-clock measurements (requires BIOS/power-play or ADL work), CI on a
  self-hosted gfx1010 runner, automatic regression gates for kernels and
  engines, and reproducible benchmark manifests (seeds, prompts, cache state).

## Next steps (ordered)

1. ZLUDA decode: support CUDA graphs or implement a persistent/merged decode
   kernel; target is to close the 4-8x gap to Vulkan at 1-8k context.
2. torch.compile for decode once graphs work; otherwise per-layer CUDA-graph
   capture in the harness.
3. Fused int8 GEMV + tiled int4 prefill; exact GPTQ/AWQ loading.
4. Long-context decode tuning (flash-decoding split count, KV layout).
5. Same-engine comparison once ZLUDA supports the llama.cpp launch API.

### llama.cpp addendum (2026-11-25, final verdict)

After the ZLUDA launch-attribute patch, llama.cpp's CUDA backend runs only in a
narrow, unusable configuration:

- `-ngl 0` (CPU compute through the CUDA backend): works (pp16 124 tok/s,
  tg2 17.4 tok/s).
- `-ngl 1 -fa 0` with the F16 GGUF: works (pp8 415.6 tok/s, tg1 61.5 tok/s).
- Increasing the number of offloaded layers (`-ngl 2..99`) either fails with
  `CUDA error: unspecified launch failure` / `operation not supported` or
  **hard-crashes the AMD driver** (the machine has to be rebooted; this happened
  several times). The Q4_K_M model (K-quant dequant/MMQ kernels) hangs even
  with `-ngl 1`.
- Workarounds have no effect: `GGML_CUDA_PDL=0`, `GGML_CUDA_DISABLE_GRAPHS=1`,
  `GGML_CUDA_NO_PINNED=1`, `GGML_CUDA_DISABLE_FUSION=1`,
  `GGML_CUDA_FORCE_MMQ=1`.

**Verdict:** running llama.cpp with GPU offload through ZLUDA is not viable on
this machine today; it is a ZLUDA runtime/compiler problem (llama.cpp's kernel
set + launch/graph model vs. ZLUDA's PTX JIT and HIP launch path), not a
llama.cpp binary issue. The patched launch-attribute support
(`zluda/src/impl/driver.rs`, `cudaLaunchKernelEx`/PDL attributes accepted,
kernels serialized) is kept because it is a general ZLUDA improvement. A
legitimate same-engine Vulkan-vs-ZLUDA comparison is therefore still not
available; `bench/llamacpp_zluda_matrix.ps1` is ready for when the ZLUDA side
can run llama.cpp kernels. **Warning: do not run llama.cpp with ngl>1 through
ZLUDA unless a reboot is acceptable - the driver can hang hard.**
