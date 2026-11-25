# Same-engine comparison: llama.cpp Vulkan vs llama.cpp on ZLUDA (gfx1010)

Date: 2026-11-25. GPU: Radeon RX 5700 XT (gfx1010, 8 GB). Model: Qwen2.5-0.5B
Instruct, **F16 GGUF** (the same file for both backends). This is the first
legitimate same-engine Vulkan-vs-ZLUDA measurement on this machine.

## Why this needed a custom llama.cpp build

The official Windows CUDA release builds of llama.cpp ship only SASS cubins for
sm_50/61/70/75/80 and no PTX. ZLUDA executes PTX only, so those builds report
`CUDA kernel ... has no device code compatible with CUDA arch 86` and either
fail or hang. The fix is a local build with compute_86 PTX:

- portable CUDA 12.4.1 (nvcc 12.4.131) extracted from NVIDIA redist components
  (`cuda_nvcc`, `cuda_cudart`, `cuda_cccl`, `libcublas`, plus a minimal
  `nvml.h` stub - NVML is loaded dynamically and is absent on AMD);
- VS 2022 BuildTools as host compiler (nvcc 12.4 rejects VS 2026);
- `cmake -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86-virtual ...`
  (`86-virtual` = PTX-only, exactly what ZLUDA JITs).

Full recipe: `docs/LLAMACPP_ZLUDA_BUILD.md`. Verified: the resulting
`ggml-cuda.dll` has `.target sm_86` only.

## Configuration

- **Vulkan**: Ollama 0.40.18 (llama.cpp Vulkan backend), 100% GPU offload,
  `OLLAMA_FLASH_ATTENTION=0`, `OLLAMA_KV_CACHE_TYPE=f16`, num_ctx up to 32768.
- **ZLUDA**: local PTX build, `-ngl 99 -fa 0`, `GGML_CUDA_PDL=0`,
  `GGML_CUDA_DISABLE_GRAPHS=1`, run through `run-zluda.cmd`.
- Flash attention is disabled on both sides: llama.cpp's FA kernels use
  `mma.sync` (tensor-core instructions) which RDNA1 does not have and ZLUDA
  cannot translate, so FA works on Vulkan but fails on ZLUDA. FA=0 uses the
  vector kernels and is the common denominator.
- Measurement: `llama-bench` (`pp` prefill, `tg` 32-token decode) for ZLUDA and
  the Ollama HTTP API (prompt/eval durations) for Vulkan; single run, GPU
  clocks cannot be fixed (+-15-30% run-to-run variance).

## Results (Qwen2.5-0.5B F16 GGUF, FA=0)

| context | ZLUDA prefill tok/s | Vulkan prefill tok/s | ZLUDA decode tok/s | Vulkan decode tok/s |
|---|---|---|---|---|
| 2048 | 6144 | 6258 | 184.2 | 212.3 |
| 8192 | 3294 | 4179 | 184.6 | 172.7 |
| 32768 | 732 | n/a (Ollama clamps F16 to 8192 ctx on 8 GB) | 183.9 | n/a |

Raw data: `bench/results/llamacpp_zluda_vs_vulkan.json`.

**Reading of the numbers:** on this small model the two backends are close -
ZLUDA is within -2% of Vulkan at 2k prefill and -21% at 8k, decode is -13% at
2k and +7% at 8k (all within/about the run-to-run spread except the 8k
prefill). ZLUDA can go to 32k context with F16 while Ollama clamps F16 to 8192
on this card, and at 32k ZLUDA still produces 732 tok/s prefill and 184 tok/s
decode.

## Limitations / not yet working on ZLUDA

- `-fa 1` (llama.cpp flash attention): fails in `fattn-mma-f16` kernels
  (tensor-core/mma path not supported by ZLUDA on RDNA1). Vulkan supports it.
- `Q4_K_M` GGUF: hangs in the quantized MMQ kernels on ZLUDA (watchdog killed
  it after 100 s of no progress; the device itself stayed healthy). F16 works.
- CUDA graphs and PDL are disabled on the ZLUDA side
  (`GGML_CUDA_DISABLE_GRAPHS=1`, `GGML_CUDA_PDL=0`); the launch-attribute patch
  in this fork is what lets llama.cpp get this far.

## Safety note

Testing llama.cpp with GPU offload through ZLUDA used to hard-crash the AMD
driver (reboot required), and a watchdog that tried `Restart-PnpDevice`
crashed Windows once. Since the PTX build works, run these tests only through
`tools/gpu_watchdog.ps1` with driver restart disabled (default):
`-PollSeconds 20..30 -StallLimit 4..8`. Do not run `-ngl > 1` with the
official SASS-only builds.

## Reproduce

```bat
set "VS=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"
call G:\zluda\build-env.cmd build_llama_ptx86.cmd
set "PATH=C:\build\llama.cpp\build-ptx86\bin;C:\cuda-12.4.1\bin;%PATH%"
set GGML_CUDA_DISABLE_GRAPHS=1
set GGML_CUDA_PDL=0
G:\zluda\run-zluda.cmd C:\build\llama.cpp\build-ptx86\bin\llama-bench.exe ^
  -m G:\zluda\models\qwen05-gguf\qwen2.5-0.5b-instruct-fp16.gguf ^
  -p 2048,8192,32768 -n 32 -ngl 99 -fa 0 -r 1
```