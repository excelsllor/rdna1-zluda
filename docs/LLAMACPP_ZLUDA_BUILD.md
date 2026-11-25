# Building llama.cpp for ZLUDA on gfx1010 (RDNA1)

## Why a custom build is needed

The official `llama-b####-bin-win-cuda-*.zip` release builds contain only SASS
cubins compiled for `sm_50/61/70/75/80` (checked in b12721, b9802, b6000 by
scanning `ggml-cuda.dll` for `.target sm_` - no 86, and no PTX source at all).
ZLUDA executes PTX and JIT-compiles it for gfx1010; SASS cannot be used. As a
result llama.cpp prints
`CUDA kernel ... has no device code compatible with CUDA arch 86` and GPU
offload fails or hangs (hard driver crashes were observed with high `-ngl`).

The fix is a local llama.cpp build that embeds **compute_86 PTX** (ZLUDA reports
compute capability 8.6 through `ZLUDA_CC`, default 8.6).

## Prerequisites (no admin rights needed)

- Visual Studio 2022 Build Tools with the C++ toolchain (VS 2026 is rejected by
  nvcc 12.4 as an unsupported host compiler, so use 2022: set `VS` before
  `build-env.cmd`).
- Portable CUDA 12.4.1 redist components (installed manually by extracting
  zips, because the CUDA installer requires admin):

```powershell
# from https://developer.download.nvidia.com/compute/cuda/redist/redistrib_12.4.1.json
# cuda_nvcc   -> nvcc, ptxas, cudafe
# cuda_cudart -> runtime headers/libs
# cuda_cccl   -> thrust/cub headers
# libcublas   -> cuBLAS headers + import libraries (llama.cpp links cuBLAS)
# extract each archive and merge its bin/include/lib/nvvm/src into C:\cuda-12.4.1
```

Verified versions: nvcc 12.4.131, VS 2022 BuildTools, CMake/Ninja bundled with
VS.

## Configure and build

```bat
set "VS=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"
call G:\zluda\build-env.cmd build_llama_ptx86.cmd
```

where `build_llama_ptx86.cmd` runs:

```bat
cd /d C:\build\llama.cpp
cmake -B build-ptx86 -G Ninja -DGGML_CUDA=ON ^
      -DCMAKE_CUDA_ARCHITECTURES=86-virtual ^
      -DGGML_NATIVE=OFF -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release ^
      -DCMAKE_CUDA_COMPILER=C:/cuda-12.4.1/bin/nvcc.exe ^
      -DCUDAToolkit_ROOT=C:/cuda-12.4.1
cmake --build build-ptx86 -j 6 --target llama-bench llama-cli
```

`86-virtual` makes CMake emit `-gencode arch=compute_86,code=compute_86`, i.e.
PTX only (exactly what ZLUDA needs). Verify after the build:

```powershell
$s = [Text.Encoding]::ASCII.GetString([IO.File]::ReadAllBytes(
     "C:\build\llama.cpp\build-ptx86\bin\ggml-cuda.dll"))
([regex]::Matches($s, '\.target sm_(\d+)') | % { $_.Groups[1].Value } | Sort-Object -Unique)
# expected: 86
```

## Running and safety

- Always run through `run-zluda.cmd` with the local build in `PATH` and the
  cudart DLLs available (`tools/llamacpp-cuda/cudart` or the new build's bin).
- Start with `-ngl 1 -fa 0 -p 8 -n 1`; scale offload only under
  `tools/gpu_watchdog.ps1` (kill-only; driver restart is disabled by default
  because it crashed Windows once).
- If the GPU wedges, reboot. Do not use `-ngl` high values without a reboot
  being acceptable.

## Status

- 2026-11-25: portable CUDA toolchain + llama.cpp build configured and started;
  see the ROADMAP entry for the outcome and ENGINE_COMPARISON.md for the
  Vulkan-vs-ZLUDA matrix once the PTX build runs.