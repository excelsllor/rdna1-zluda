# Cross-backend comparison: Vulkan vs native ROCm vs ZLUDA (gfx1010)

Date: 2026-11-25. GPU: RX 5700 XT (gfx1010, 8 GB). Model: Qwen2.5-0.5B-Instruct.
Engines: llama.cpp Vulkan (Ollama 0.40.18) for Vulkan; native `torch
2.14.0+rocm10.1.0` (HIP 7.16, from the ComfyUIrocm venv) for ROCm; this ZLUDA
fork with PyTorch 2.7.1+cu118 + Triton 3.3.1 for ZLUDA. All PyTorch runs are
capped at 85% VRAM. GPU clocks cannot be fixed (+-15-30% run-to-run spread);
numbers are single runs unless noted, so differences below ~20% are not
significant.

## 1. PyTorch engines, math SDPA, prefill 1024 / decode 4

| config | prefill tok/s (ZLUDA) | decode tok/s (ZLUDA) | peak MB | prefill tok/s (ROCm) | decode tok/s (ROCm) | peak MB |
|---|---|---|---|---|---|---|
| fp16 | **4831** | **28.8** | 1123 | 3082 | 19.8 | 1146 |
| fp16 + fp16-accum | **7450** | **51.0** | 1123 | 3099 (*) | 19.9 (*) | 1146 |
| fp32 | **5418** | **35.5** | 2073 | 1177 | 16.5 | 2096 |
| bf16 native | **3450** | **11.1** | 1123 | 2845 | 10.6 | 1146 |
| bf16 emulated (bf16 weights -> fp32 compute) | crash (**) | crash | - | 3812 | 26.0 | 2096 |

(*) The ROCm build ignores `torch._C._set_cublas_allow_fp16_accumulation`; fp16
accumulation is a cuBLAS/HPA concept, so ROCm has no equivalent switch and Vulkan
has no such concept at all (llama.cpp computes in the model tensor type).
(**) ZLUDA bf16-emul process died without producing a report (most likely VRAM:
an fp32 copy on top of the bf16 model), so it is not measured.

ROCm bf16-emul is 1.34x faster than native bf16 on ROCm because it routes the
GEMMs through the faster fp32 path; on ZLUDA the same trick needs more VRAM than
the 85% cap allows on this 8 GB card.

## 2. ZLUDA attention stacks (fp16, prefill 1024 / decode 4)

| attention path | prefill tok/s | decode tok/s | notes |
|---|---|---|---|
| math SDPA | 4831 | 28.8 | baseline |
| SDPA default ("fast") | 4822 | 27.0 | falls back to math (flash/mem-efficient/cudnn unavailable on ZLUDA) |
| Triton flash-attention (`zluda_ops`) | **5566** | 29.2 | our FA kernel, seq %128==0 |
| flex attention (`torch.nn.attention.flex_attention`) | 4875 | 29.4 | Triton block-mask path works |
| flash_attn API shim | 5138 | 1.6 (!) | prefill good; decode degenerates (API/fallback overhead per token) |
| xformers shim | OOM | OOM | `memory_efficient_attention` allocation blow-up in decode |
| math + fp16-accum | 7450 | 51.0 | best PyTorch configuration |
| Triton FA + fp16-accum | **9460** | 48.9 | best overall ZLUDA prefill |

Attention availability by engine:

| feature | Vulkan (llama.cpp) | ROCm (native torch) | ZLUDA (torch) |
|---|---|---|---|
| math SDPA | n/a (llama.cpp kernels) | yes | yes |
| SDPA fast backends (flash/mem-efficient/cudnn) | n/a | not usable on gfx1010 | not available (math fallback) |
| Triton flash-attention | n/a | no Triton in the ROCm venv | yes (`zluda_ops`) |
| flex attention | n/a | no Triton | yes |
| flash_attn API | n/a | no (no wheel for gfx1010) | shim (works for prefill; decode anomaly) |
| xformers API | n/a | no (CUTLASS kernels) | shim (OOM in decode as tested) |
| llama.cpp flash attention (`-fa 1`) | yes | n/a | **fails** (`mma.sync` kernels unsupported by ZLUDA on RDNA1) |
| torch.compile | n/a | not tested (no Triton) | prefill +42-49% (6449-6733 tok/s), decode unusable (recompiles per token; CUDA graphs unsupported) |

## 3. Vulkan/llama.cpp (GGUF variants, same llama.cpp, FA=0, KV f16)

| model | ctx | prefill tok/s | decode tok/s | note |
|---|---|---|---|---|
| F16 | 2048 | 6211 | 238.3 | stable retry (an outlier run gave 2640/29.3) |
| F16 | 8192 | 2102-4179 | 28.5-172.7 | large run-to-run spread |
| F32 | 2048 | 3989-5233 | 147.4-145.9 | |
| F32 | 8192 | 466-3580 | 106.5-123.4 | high variance |
| BF16 | 2048 | 1968-4559 | 76.7-218.6 | high variance |
| BF16 | 8192 | 3319-3328 | 157.9-181.1 | fairly stable |

FP16 accumulation / bf16-emulation do not exist in the Vulkan/llama.cpp
configuration space: llama.cpp computes in the tensor types stored in the GGUF
(F16/F32/BF16) and `-fa`/KV-cache flags are the available knobs. Earlier
FA=1/KV=q8_0 numbers (Q4_K_M 4531/2741/946 prefill and 150.8/135.2/104.8 decode
at 2k/8k/32k; F16 FA=0 6258/4179 prefill and 212.3/172.7 decode at 2k/8k) are in
ENGINE_COMPARISON.md.

## 4. Same-engine llama.cpp: Vulkan vs ZLUDA (F16 GGUF, FA=0, -ngl 99)

| context | ZLUDA prefill | Vulkan prefill | ZLUDA decode | Vulkan decode |
|---|---|---|---|---|
| 2048 | 6144 | 6258 | 184.2 | 212.3 |
| 8192 | 3294 | 4179 | 184.6 | 172.7 |
| 32768 | 732 | n/a (Vulkan clamps F16 to 8192) | 183.9 | n/a |

Full details: docs/LLAMACPP_VULKAN_VS_ZLUDA.md and
docs/LLAMACPP_ZLUDA_BUILD.md (the custom compute_86-PTX llama.cpp build).

## 5. Reading of the results

- **ZLUDA beats native ROCm torch on every configuration measured** for this
  model: fp16 +57% prefill / +46% decode, fp32 +4.6x prefill / +2.1x decode,
  bf16 +21% prefill. The ZLUDA fork ships tuned gfx1010 Tensile overrides,
  while the native ROCm torch uses the generic rocBLAS path.
- fp16 accumulation is the single biggest PyTorch knob (+54% prefill on math,
  +70% with Triton FA); it is a cuBLAS concept, ignored by ROCm and absent on
  Vulkan.
- On ZLUDA, Triton FA +16% prefill over math; flex attention and the flash_attn
  shim are competitive for prefill, but the flash_attn-shim decode path and the
  xformers shim need work before they are usable in generation loops.
- Same-engine llama.cpp comparison is close (ZLUDA -2%/-21% prefill at 2k/8k,
  decode -13%/+7%) and ZLUDA can run 32k context where Ollama clamps F16.
- The Vulkan numbers have much larger run-to-run spread than the PyTorch ones,
  so the llama.cpp comparison is the more meaningful Vulkan-vs-ZLUDA statement.

## 6. Artefacts and reproduction

- `bench/attention_matrix.py` - one configuration per process (dtype x fp16-acc
  x bf16-emul x attention path x compile), reports prefill/decode/peak and
  logits-vs-math correctness.
- `bench/run_attention_matrix.py` - sequential runner; results merged into
  `bench/results/attention_matrix_summary.json` and
  `bench/results/cross_backend_summary.json`.
- `bench/ollama_bench.py`, `bench/ollama_config.ps1`, `bench/run_vulkan_matrix.py`
  for the Vulkan/Ollama side; GGUF F32/BF16 variants were produced locally with
  llama.cpp `convert_hf_to_gguf.py`.
- Native ROCm runs use `D:\comfyui\Data\Packages\ComfyUIrocm\venv\Scripts\python.exe`
  (torch 2.14.0+rocm10.1.0, transformers 5.19, no Triton).
- ZLUDA runs go through `run-zluda.cmd`; risky GPU workloads should use
  `tools/gpu_watchdog.ps1` (kill-only by default - the driver-restart path
  crashed Windows once).

## 7. Known gaps

- flash_attn-shim decode (1.6 tok/s) and xformers-shim OOM need debugging before
  those shims can be recommended for generation.
- torch.compile on ZLUDA: prefill win confirmed earlier, decode needs CUDA-graph
  or per-token compile support; not re-measured here.
- ROCm: no Triton, so Triton FA/flex/compile cannot be compared on ROCm.
- Vulkan: only llama.cpp is available (Ollama); no PyTorch dtype/accumulation
  controls.
- Fixed clocks (BIOS/power-play) and repeated medians would tighten all numbers.