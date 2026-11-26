# Compatibility matrix: llama.cpp, torch.compile, attention paths, ComfyUI

Date: 2026-11-26. GPU: RX 5700 XT (gfx1010, 8 GB), ZLUDA fork as of this repo.
Each attention path was run in a separate process because crash-prone backends
poison the CUDA context for the whole process.

## 1. Attention paths (fp16, B=1, H=8, D=64)

### Prefill: S=256, causal

| path | status | max err vs math | time |
|---|---|---|---|
| math SDPA (MATH backend) | **OK** | 0 | 0.266 ms |
| `sdpa_default` (PyTorch chooses backend) | **FAIL** | - | `unspecified launch failure` + `hipModuleLoadData ... 719` |
| `sdpa_flash` (FLASH_ATTENTION) | **FAIL** | - | `no available kernel` / launch failure |
| `sdpa_efficient` (EFFICIENT_ATTENTION) | **FAIL** | - | `unspecified launch failure` |
| `sdpa_cudnn` (CUDNN_ATTENTION) | **FAIL** | - | `FATAL: kernel fmha_cutlass...sm80 is for sm80-sm100` |
| Triton FA (`zluda_ops.flash_attention`) | **OK** | 0.00195 | 0.228 ms |
| flex attention (`torch.nn.attention.flex_attention`) | **OK** | 0.00195 | 3.93 ms (slow: block-mask path) |
| flash_attn shim (Triton FA backend) | **OK** | 0.00195 | 0.271 ms |
| xformers shim (Triton FA backend) | **OK** | 0.00195 | 0.274 ms |

### Decode: GQA 14 q / 2 kv heads, q_len=1, KV=1024, non-causal

| path | status | max err | time |
|---|---|---|---|
| math SDPA | **OK** | 0 | 0.305 ms |
| `sdpa_default` | **OK** | 0 | 0.277 ms (falls back to math) |
| Triton flash-decoding (GQA-aware) | **OK** | 6.1e-5 | **0.121 ms** (2.5x faster) |
| flash_attn shim (flash-decoding) | **OK** | 6.1e-5 | **0.113 ms** |
| xformers shim (flash-decoding) | **OK** | 6.1e-5 | 0.154 ms |

Rule: PyTorch's flash/efficient/cudnn SDPA backends are unusable under ZLUDA
(crash or wrong-arch cutlass fmha). `zluda_ops.auto` now disables them globally
and keeps MATH as the fallback; Triton FA (prefill) and flash-decoding (decode)
are the fast paths. Decode attention is 2.5x faster with flash-decoding; flex
attention works but is ~17x slower than Triton FA for prefill.

## 2. llama.cpp (custom compute_86-PTX build, full -ngl 99 offload)

| model / flags | status | result |
|---|---|---|
| F16, `-fa 0` | **OK** | pp64 1762 tok/s, tg4 172 tok/s |
| F32 GGUF, `-fa 0` | **OK** | pp64 2841 tok/s, tg4 139 tok/s (F32 prefill is faster than F16 because fp32 GEMM is faster on gfx1010) |
| F16, `-fa 1` | **FAIL** | `CUDA error: named symbol not found` (fatnn/mma kernels) |
| Q4_K_M / Q8_0 (MMQ quant kernels) | **not testable safely** | Q4_K_M hangs the driver on this stack (requires reboot); Q8_0 uses the same MMQ kernel family - do not run without reboot readiness |

Official SASS-only llama.cpp builds remain unusable (no PTX for sm_86); the
custom build with `-DCMAKE_CUDA_ARCHITECTURES=86-virtual` is the supported path
(docs/LLAMACPP_ZLUDA_BUILD.md).

## 3. torch.compile

Raw `torch.compile(model)` fails while tracing SDPA: PyTorch selects the cuDNN
attention path during compilation, which hits
`FATAL: kernel fmha_cutlassF_f16... is for sm80-sm100` and aborts.
With MATH-only SDPA (the safe configuration; e.g. the `--attn math` wrapper in
bench/model_bench.py) compile succeeds for prefill:

| config | status |
|---|---|
| prefill compile (`mode=default`/`max-autotune-no-cudagraphs`, math SDPA) | **OK** - 6449-6733 tok/s prefill (vs 4525 eager) |
| prefill compile with raw SDPA backend selection | **FAIL** - cuDNN fmha arch fatal |
| prefill compile with `zluda_ops.auto` imported | **FAIL** - `InductorError: Triton Error [CUDA]: error` |
| decode compile with StaticCache | **FAIL** - Inductor kernel generation (`Triton Error [CUDA]`) |
| decode compile with DynamicCache | unusable - recompiles per token (0.27 tok/s measured earlier) |
| `mode=reduce-overhead` (CUDA graphs in Inductor) | **FAIL** - graph capture unsupported |

Working recipe: use `torch.compile` for **prefill only**, with math SDPA (or
Triton FA through zluda_ops but *outside* the compiled region), and keep decode
on CUDA-graph replay (`zluda_ops.graph_decode.GraphDecoder`).

## 4. ComfyUI (target resolution 0.5 MP = 704x704)

- Base-mode launcher (`comfy-zluda.cmd`) starts python and imports ComfyUI but
  **did not reach the HTTP API within 200 s** under ZLUDA (process alive, no
  error in the log) - startup is either very slow or stuck in an import.
- Custom-nodes launcher (`comfy-zluda-nodes.cmd`) fails during node startup:
  `comfyui-sam3` prestartup script raises `No module named 'comfy_env'`, and
  the startup does not complete.
- Available assets: Flux-like GGUF `unstableRevolutionF2K_AlphaF2K4BQ51.gguf`
  (2.87 GB), `flux2-vae.safetensors`, Qwen3-VL GGUF text encoders; custom nodes
  for GGUF/Flux/attention exist under `D:\comfyui\Data\CustomNodes`.
- e2e 0.5 MP generation: **not completed in this cycle**. Blocker: ComfyUI
  startup under ZLUDA + the Flux GGUF pipeline requires its matching text
  encoders and more VRAM than a 8 GB card has together with the runtime.
  Plan: after a fresh reboot, launch with only the required nodes
  (ComfyUI-GGUF + flux2k), 704x704, 1-4 steps, and measure.

## 5. Summary of what does NOT work (documented)

1. PyTorch fast SDPA backends (flash/efficient/cudnn) -> crash / wrong-arch kernels.
2. llama.cpp `-fa 1` (mma/fatnn kernels) and quantized MMQ models (Q4/Q8 hang).
3. torch.compile for decode (StaticCache) and reduce-overhead (CUDA graphs).
4. ComfyUI startup / Flux GGUF e2e under ZLUDA (not yet achieved).
5. fp64 BLAS (cublasDgemm), cublasDgemv (fp64 only; fp32/fp16 GEMV work).
6. CUPTI/torch profiler traces are empty; GPU clocks cannot be fixed (ADL).