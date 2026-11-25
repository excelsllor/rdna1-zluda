# Engine comparison: Vulkan (Ollama) vs ZLUDA (PyTorch/Triton) on gfx1010

Updated: 2026-11-25. GPU: RX 5700 XT (gfx1010, 8 GB). Model: Qwen2.5-0.5B
(same weights: GGUF for Ollama, safetensors for PyTorch). VRAM capped to 85%
for PyTorch runs. GPU clocks cannot be fixed: single-run spread is +-15-30%.

## Setups

- **Vulkan**: Ollama 0.40.18, llama.cpp Vulkan backend, 100% GPU offload.
  GGUF Q4_K_M and F16 imported with num_ctx 32768.
  `OLLAMA_FLASH_ATTENTION` (on/off) and `OLLAMA_KV_CACHE_TYPE` (q8_0/f16/f32)
  set through `bench/ollama_config.ps1`; measurement via `bench/ollama_bench.py`
  (`bench/run_vulkan_matrix.py` for the matrix).
- **ZLUDA**: this fork (PyTorch 2.7.1+cu118, Triton 3.3.1, ROCm 10.1 + tuned
  rocBLAS 7.1.1), benchmarked with `bench/model_bench.py`.
- **llama.cpp CUDA under ZLUDA: blocked.** b12721 CUDA 12.4 build loads under
  ZLUDA (device found: 8176 MiB, compute 8.6) but kernel launch fails with
  `CUDA error: operation not supported` at `cudaLaunchKernelEx` (programmatic
  dependent launch). With `GGML_CUDA_PDL=0` the run hangs. Same-engine
  Vulkan-vs-ZLUDA is therefore not possible today.
- **bitsandbytes: excluded.** 0.50.2 imports and ships ROCm libs, but its
  kernels fail under ZLUDA (`named symbol not found`, both default and
  `BNB_BACKEND=rocm`). Substitute: `zluda_ops/optim8bit.py` (Triton
  blockwise-int8 AdamW): step 0.124 ms vs 0.566 ms torch AdamW at 1 Mi
  params (4.6x), optimizer-state memory 3.94x smaller, 11% parameter deviation
  after 8 steps (int8 state quantization, same tradeoff as bnb).

## Vulkan results (Ollama, KV=q8_0, FA=1 unless noted)

| model | ctx | prompt tok | prefill tok/s | decode tok/s |
|---|---|---|---|---|
| Q4_K_M | 2048 | 1850 | 4531 | 150.8 |
| Q4_K_M | 8192 | 7310 | 2741 | 135.2 |
| Q4_K_M | 32768 | 29150 | 946 | 104.8 |
| F16 | 2048 | 1850 | 4771 | 133.7 |
| F16 | 8192 | 7310 | 2810 | 123.3 |
| F16 | 32768 (clamped to 8192) | 4098 | 3843 | 129.0 |

FA on/off and KV variants (KV=f16, FA=0 is the valid combination; q8_0 KV
requires FA, otherwise Ollama returns HTTP 500):

| model | FA | KV | ctx | prefill tok/s | decode tok/s |
|---|---|---|---|---|---|
| Q4_K_M | 1 | f16 | 2048 | 4364 | 31.8 (*) |
| Q4_K_M | 1 | f16 | 8192 | 2769 | 143.1 |
| F16 | 1 | f16 | 2048 | 4905 | 142.8 |
| F16 | 1 | f16 | 8192 | 2857 | 133.4 |
| Q4_K_M | 0 | f16 | 2048 | 2782 | 196.2 |
| Q4_K_M | 0 | f16 | 8192 | 3986 | 178.6 |
| F16 | 0 | f16 | 2048 | 6258 | 212.3 |
| F16 | 0 | f16 | 8192 | 4179 | 172.7 |

(*) outlier, likely a warm-up artefact; decode is otherwise 130-212 tok/s.

Observations: flash attention mainly helps prefill at short context; FA=0
decodes faster here; **F16 model is clamped by Ollama to 8192 context** on
8 GB while Q4_K_M reaches 32768.

## ZLUDA/PyTorch results (prefill 1024 tokens, decode 8, ctx 1024)

| config | prefill tok/s | decode tok/s | peak VRAM |
|---|---|---|---|
| fp16, math SDPA | 4525 | 22.1 | 1270 MB |
| fp16, fp16-accum | **7774** | 26.2 | 1270 MB |
| fp16, Triton FA (`--exact-seq 1024`) | 4813 | 27.9 | 1281 MB |
| bf16, math SDPA | 2798 | 10.2 | 1270 MB |
| fp32, math SDPA | 4984 | 25.3 | 2518 MB |
| int4 weights (fused dequant), fp16 | 2031 | 25.4 | 2014 MB (**) |
| int8 weights (dequant-once), fp16 | 1990 | 18.9 | 2496 MB (**) |
| 8-bit AdamW step (1 Mi fp32 params) | 0.124 ms/step vs 0.566 torch | - | 4x less state |

(**) peak includes model load + quantization temporaries; int4 should end lower
than fp16 once the guard/measurement is refined.

## Comparison and diagnosis

- **Prefill** is competitive: ZLUDA fp16 4525 tok/s vs Vulkan 4364-4905 tok/s;
  fp16 accumulation pushes ZLUDA to 7774 tok/s (~+70% over Vulkan), and fp32
  (4984) is faster than fp16 because rocBLAS fp32 (8.16 TF) beats HPA fp16
  (6.9 TF) for prefill-sized GEMMs.
- **Decode is 4-8x slower**: ZLUDA 19-28 tok/s vs Vulkan 105-212 tok/s. Main
  causes: one Python/kernel-launch-heavy forward per token, math SDPA for the
  1-token query in the default path, no CUDA graphs, no persistent/merged
  decode kernels. This is the biggest known ZLUDA-side gap.
- **Quantization** does not help prefill on ZLUDA (dequant ALU is no longer
  hidden at M>=1024) - int4 2031 / int8 1990 tok/s vs 4525 fp16; its value is
  memory/storage (see the quant sections of RESULTS_RDNA1.md). The int4 memory
  measurement still needs a guard-aware peak fix.
- **fp16-accum** is the cheapest large prefill win (7.8k tok/s) at the cost of
  accuracy (documented in RESULTS_RDNA1.md); it also slightly helps decode.
- Anthropic-tokenizer note: initial FA test used a 1021-token prompt (not
  divisible by 128) and silently fell back to math; `--exact-seq` pads to an
  exact 1024-token random prompt so the Triton path runs.

## Bugs found and fixed during this comparison

- `zluda_ops.flash_attention` fallback called the globally patched
  `F.scaled_dot_product_attention`, causing infinite recursion when the Triton
  FA path falls back (e.g. sequence not divisible by 128). Now it captures the
  original function at import (`_ORIGINAL_SDPA`).
- `torch.compile` and long-context ZLUDA runs are still to be measured
  (compile of the full model is expected to be slow on this stack).

## Next steps

1. ZLUDA decode optimizations: CUDA/HIP graphs, fused per-token kernels,
   flash-decoding dispatch in the default path, batching - target is to close
   the 4-8x gap to Vulkan.
2. Max-context ZLUDA runs with Triton FA (8k/16k/32k) and the F16-context clamp
   investigation on the Vulkan side.
3. torch.compile runs (fp16, fp16-accum) and the TORCH_LOGS fusion audit;
   implement the missing fusions.
4. Refine quantization memory accounting (peak during load/quant vs steady
   state) and re-measure int4/int8.
5. Final analysis of usage problems and missed optimizations.