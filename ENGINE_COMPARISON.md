# Engine comparison: Vulkan (Ollama) vs ZLUDA (PyTorch/Triton) on gfx1010

Status: in progress (2026-11-25). GPU: RX 5700 XT (gfx1010, 8 GB).
Model: Qwen2.5-0.5B-Instruct (same weights in GGUF for Ollama and safetensors
for PyTorch).

## Setups

- **Vulkan**: Ollama 0.40.18, llama.cpp Vulkan backend, GGUF models imported
  with `num_ctx` up to 32768. `OLLAMA_FLASH_ATTENTION=1`,
  `OLLAMA_KV_CACHE_TYPE=q8_0` (configurable through `bench/ollama_config.ps1`),
  100% GPU offload confirmed by `ollama ps`.
- **ZLUDA**: fork engine (this repo) with PyTorch 2.7.1+cu118, Triton 3.3.1,
  ROCm 10.1 user space + tuned rocBLAS 7.1.1, launcher `run-zluda.cmd`.
  Harness: `bench/model_bench.py` (fp32/fp16/bf16 x fp16-accum x math/Triton-FA
  x torch.compile x int8/int4 quantization).
- **llama.cpp CUDA under ZLUDA (blocked)**: the Windows CUDA build
  (b12721, CUDA 12.4) loads under ZLUDA, sees the RX 5700 XT (8176 MiB,
  compute 8.6), but kernel launch fails with `CUDA error: operation not
  supported` at `cudaLaunchKernelEx` (programmatic dependent launch, PDL).
  Disabling PDL via `GGML_CUDA_PDL=0` did not produce a usable run (hang).
  Conclusion: the same-engine (llama.cpp) Vulkan-vs-ZLUDA comparison is not
  currently possible; the ZLUDA side uses the PyTorch/Triton stack instead.
- **bitsandbytes (excluded)**: 0.50.2 imports and ships ROCm libraries, but its
  ops fail on ZLUDA (`named symbol not found` in its nvcc-built kernels, both
  default and `BNB_BACKEND=rocm`). Substitute implemented in this repo:
  `zluda_ops/optim8bit.py` (Triton blockwise-int8 AdamW).

## First results (Vulkan/Ollama, Q4_K_M and F16 GGUF, FA=1, KV=q8_0)

| model | context | prompt tokens | prefill tok/s | decode tok/s |
|---|---|---|---|---|
| Q4_K_M | 2048 | 1850 | 4474 | 151.4 |
| Q4_K_M | 32768 | 29150 | 943.8 | 105.4 |
| F16 | 2048 | 1850 | 3760.8 | 133.3 |
| F16 | 32768 | 4098 (*) | 3838.7 | 129.4 |

(*) F16 run reported `context_length=8192` in `ollama ps` and truncated the
prompt to ~4k tokens, i.e. Ollama clamped the context for the F16 model on this
8 GB card. Investigating (likely a VRAM-aware clamp; Q4_K_M kept 32k).

## Notes

- Vulkan equivalents of the PyTorch flags: no fp32/bf16 model execution (GGUF
  tensors are F16/Q4_K_M...), no `torch.compile`, no `fp16accum`; flash
  attention is a llama.cpp flag (`OLLAMA_FLASH_ATTENTION`), KV cache type is
  `OLLAMA_KV_CACHE_TYPE` (f16/f32/q8_0).
- All numbers are single-run medians of the Ollama API durations; GPU clocks
  cannot be fixed on this card, so +/-15-30% run-to-run variance applies.
- The ZLUDA-side measurement matrix and the final analysis (problems, missed
  optimizations) are still to be filled in.