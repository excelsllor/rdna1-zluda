# RDNA1 (gfx1010) build and usage guide

This document describes how to build ZLUDA for Radeon RX 5000 series (gfx1010)
on Windows, set up the ROCm/HIP runtime and PyTorch environment, and run
training/inference workloads. Measurements are in `RESULTS_RDNA1.md`; plans and
priorities are in `ROADMAP.md`.

## 1. Prerequisites

- Windows 10 22H2 or newer, x64.
- AMD Radeon RX 5000 series GPU with a recent Adrenalin driver.
- Visual Studio Build Tools with the MSVC C++ toolchain (VS 2022 or VS 2026).
  The bundled CMake and Ninja are sufficient.
- Rust stable (1.85+; tested with 1.96).
- Git 2.40+ and Git LFS.
- Python 3.10 (for the PyTorch environment).
- A ROCm 10.1 user space for gfx1010 (see section 4): HIP 7.16, MIOpen,
  rocFFT and rocSPARSE from the TheRock gfx1010 packages, with rocBLAS replaced
  by the tuned 7.1.1 build.

## 2. Clone

```bat
git lfs install
git config --global core.longpaths true
git clone --recursive https://github.com/excelsllor/rdna1-zluda.git ZLUDA
cd ZLUDA
git lfs pull
```

If you build from upstream instead, use tag `v7-preview.21` and apply
`local-patches/zluda-rdna1-gfx1010.patch`:

```bat
git clone --recursive --depth 1 --branch v7-preview.21 https://github.com/vosen/ZLUDA.git .
git apply local-patches\zluda-rdna1-gfx1010.patch
```

Verify that the LFS files are real (not 132-byte stubs):

- `llvm_zluda\src\device-libs\ockl.bc` starts with `BC`
- `zluda\bin\nvcudart_hybrid64.dll` starts with `MZ` and is about 1 MB

## 3. Build

Use `build-env.cmd` (sets up `vcvars64.bat`, CMake and Ninja from Visual Studio)
or run the equivalent commands manually.

```bat
build-env.cmd cargo xtask zip --release
```

Result:

- `target\release\` — `nvcuda.dll`, `zluda.exe`, `zluda64_server.exe`,
  `cublas*64_*.dll`, `cublasLt*`, `cudnn64_8/9.dll`, `cufft*`, `cusparse*`,
  `nvml.dll`, `nvapi64.dll`, `nvcudart_hybrid64.dll`, `cuda_check.exe`,
  trace shims.
- `target\release\zluda.zip` — packaged build.

Notes:

- The build compiles LLVM (AMDGPU + lld) from `ext/llvm-project`; expect tens of
  minutes and several GB of disk space.
- The local patch disables LLVM's DIA/PDB support (`LLVM_ENABLE_DIA_SDK=OFF`)
  because the Visual Studio Build Tools installation may not include ATL.
- Do not set `HIP_PATH` for the build; it is only needed at runtime.

## 4. Runtime stack

ZLUDA loads the HIP runtime and performance libraries at run time. This fork
supports a single stack.

### 4.1 ROCm 10.1 + tuned rocBLAS 7.1.1

- ROCm 10.1 user space from the TheRock gfx1010 packages (HIP 7.16, MIOpen,
  rocFFT, rocSPARSE). The easiest source is the PyTorch ROCm environment
  (`torch ... +rocm10.1`) or the `rocm_sdk_*` packages.
- rocBLAS 7.1.1 with tuned gfx1010 Tensile libraries from the
  `likelovewant/ROCmLibs-for-gfx1103-AMD780M-APU` `v0.7.1.1` release
  (`gfx1010-xnack-gfx1011-xnack-gfx1012-xnack-.7z`).

Staging:

1. Create a runtime root with a `bin` directory.
2. Copy the HIP core DLLs (`amdhip64_7.dll`, `amd_comgr.dll`, `rocm_kpack.dll`,
   `hiprtc0716.dll`, `hiprtc-builtins0716.dll`, `rocm-openblas*.dll`,
   `amdocl64.dll`, `cltrace.dll`) and all performance library DLLs
   (`rocblas.dll`, `MIOpen.dll`, `libhipblaslt.dll`, `rocfft.dll`,
   `hipfft.dll`, `rocsparse.dll`, `rocsolver.dll`, `rocrand.dll`, `hipblas.dll`,
   `hipsolver.dll`, `hipsparse.dll`, `hipdnn_backend.dll`, `origami.dll`) into
   `<runtime>\bin`. Copy `rocblas\library` and `.kpack`.
3. Back up the stock `rocblas.dll` and `rocblas\library`.
4. Replace `rocblas.dll` with the 7.1.1 dll from the tuned package and put the
   package's `library` contents into `<runtime>\bin\rocblas\library`.
5. Verify: `run-zluda.cmd cuda_check.exe` — every line must be `OK`
   (`nvapi` reports “Library check not implemented” and is informational).

Notes:

- ROCm 10.1's stock rocBLAS only contains fallback Tensile kernels for gfx1010;
  the tuned package adds real kernels (SS/HH/HPA) and roughly doubles GEMM
  throughput.
- rocBLAS validates Tensile libraries against its own version; the tuned
  libraries must match the shipped `rocblas.dll`, so both are replaced together.
- Keep the stock files as backup; new ROCm updates can overwrite them.

## 5. PyTorch environment

```bat
python -m venv venv
venv\Scripts\python.exe -m pip install --upgrade pip
venv\Scripts\python.exe -m pip install torch==2.7.1+cu118 ^
    --index-url https://download.pytorch.org/whl/cu118
venv\Scripts\python.exe -m pip install numpy triton-windows==3.3.1.post21
```

Why `cu118`: PyTorch CUDA wheels for 11.8 ship `compute_37` PTX, which ZLUDA
consumes. `cu128`/`cu13x` wheels contain only SASS and cannot be used by ZLUDA.

Triton must produce PTX, not NVIDIA SASS; point it at ZLUDA's `ptxas.exe` stub
(see environment variables below).

## 6. Environment variables and launchers

`run-zluda.cmd` sets:

```bat
set HIP_PATH=<runtime root>
set HIP_VISIBLE_DEVICES=0
set ZLUDA_CC=8.6
set TRITON_PTXAS_PATH=target\release\ptxas.exe
set TRITON_CACHE_DIR=<repo>\.triton_cache
set TORCHINDUCTOR_CACHE_DIR=<repo>\.inductor_cache
set PATH=target\release;<runtime root>\bin;%PATH%
target\release\zluda.exe -- %*
```

Important:

- `ZLUDA_CC=8.6`: the default reported compute capability 8.5 is not a valid
  NVIDIA SM and is rejected by Triton's `ptxas`.
- `TRITON_PTXAS_PATH`: Triton's bundled `ptxas` would create NVIDIA cubins;
  ZLUDA needs PTX.
- Keep Triton/Inductor caches isolated from other installations (e.g. native
  ROCm ComfyUI environments).
- `DISABLE_ADDMM_CUDA_LT=1` is optional: it forces PyTorch's `linear`/`addmm`
  through plain cuBLAS instead of cuBLASLt. The ZLUDA cuBLASLt fallback works
  without it.

## 7. Running workloads

```bat
run-zluda.cmd venv\Scripts\python.exe train.py
run-zluda.cmd venv\Scripts\python.exe infer.py
```

In Python:

```python
import torch
print(torch.cuda.get_device_name(0))          # "AMD Radeon RX 5700 XT [ZLUDA]"
a = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
b = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
c = a @ b
```

### fp16 accumulation (inference only)

```python
torch._C._set_cublas_allow_fp16_accumulation(True)  # selects packed HH kernels
```

- Doubles fp16 GEMM throughput (about 12.3 TF at 4096), but reduces accuracy
  (max abs error grows with K). Use only when validated for your model.
- Training should keep fp32 accumulation (default).

### Attention

- PyTorch SDPA `flash` and `memory_efficient` backends are unavailable on
  gfx1010 (kernels compiled only for MFMA architectures). Use the math backend.
- Triton flash attention (forward+backward) is available in
  `bench/triton_flash_canon.py`; gradients match math SDPA.
- FlexAttention (`torch.nn.attention.flex_attention` with `torch.compile`)
  works and is faster than math SDPA at small sizes.
- `xformers` and `flash_attn` packages are not available for this PyTorch build
  on Windows; see `ROADMAP.md` for the compatibility workstream.

### ComfyUI

Verified with ComfyUI 0.45.0 and a separate Python 3.12 venv
(`venv-comfy`, `torch==2.7.1+cu118`, `torchvision==0.22.1+cu118`,
`triton-windows==3.3.1.post21`); the native ROCm venv is left untouched.

```bat
rem base mode, no custom nodes (most reliable)
start-comfy.cmd

rem full mode with custom nodes (UTF-8 is required: some nodes raise
rem ImportError messages containing emoji and crash a cp1251 console)
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
comfy-zluda-nodes.cmd
```

- Both launchers use `--use-pytorch-cross-attention` (math SDPA),
  `--force-fp16`, `--disable-smart-memory`, `--gpu-only`,
  `--disable-pinned-memory` (8 GB VRAM).
- Do not expect PyTorch's flash SDPA or the `flash_attn` package to work on
  gfx1010; use math SDPA or the Triton flash implementation from
  `bench/triton_flash_canon.py`.
- Custom nodes start only when their Python dependencies are installed in
  `venv-comfy`. In a core-only install, nodes such as VideoHelperSuite,
  controlnet_aux, easy-use, BrushNet, teacache, ultraflux, was-node-suite,
  utils-nodes and sam3 fail to import. Use ComfyUI-Manager to install the
  dependencies you need, or disable those nodes.
- GGUF support (ComfyUI-GGUF) needs:

  ```bat
  venv-comfy\Scripts\python.exe -m pip install "gguf>=0.13.0" sentencepiece protobuf
  ```

  After installation the `UnetLoaderGGUF`/`DualCLIPLoaderGGUF` nodes load
  `.gguf` models from the `diffusion_models` search paths.
- Reuse an existing ComfyUI model library without copying files: point the
  `extra_model_paths.yaml` entries (`checkpoints`, `diffusion_models`,
  `text_encoders`, `loras`, `vae`, ...) at the existing model root. The
  ZLUDA instance then shares the same checkpoints, GGUF files and LoRAs.
- If `torch.cuda.is_available()` starts hanging and even a plain HIP
  `hipStreamCreateWithFlags` blocks, the AMD driver is wedged (usually after
  force-killing a process during CUDA init). Restart the display device, e.g.
  `pnputil /restart-device "<GPU instance id>"` from an elevated shell, or
  reboot. Never force-kill ComfyUI while it is initializing CUDA unless
  necessary.

## 8. Validation and benchmarks

```bat
run-zluda.cmd target\release\cuda_check.exe
run-zluda.cmd venv\Scripts\python.exe smoke\torch_check.py
run-zluda.cmd venv\Scripts\python.exe smoke\triton_check.py
run-zluda.cmd venv\Scripts\python.exe bench\matmul_bench.py
run-zluda.cmd venv\Scripts\python.exe bench\fp16_acc.py
run-zluda.cmd venv\Scripts\python.exe bench\triton_flash_canon.py
run-zluda.cmd venv\Scripts\python.exe bench\math_attn_fb.py
run-zluda.cmd venv\Scripts\python.exe smoke\flex_attn.py
run-zluda.cmd venv\Scripts\python.exe bench\roofline.py
run-zluda.cmd venv\Scripts\python.exe bench\regression.py
run-zluda.cmd venv\Scripts\python.exe bench\gemv.py
```

Expected reference numbers are in `RESULTS_RDNA1.md`.

## 9. Troubleshooting

- **Build fails in `LVMDebugInfoPDB` / `atlbase.h`**: the patch disables
  `LLVM_ENABLE_DIA_SDK`. If you build an unpatched tree, install the ATL
  component or apply the patch.
- **Build fails with “git lfs stub”**: run `git lfs install` and `git lfs pull`.
- **`Unknown symbol` / `Cannot select` for `gfx1010:xnack-`**: apply the
  `gcn_arch` feature-stripping patch.
- **Fatal LLVM abort for `dp4a`**: apply the `dp4a` software fallback patch.
- **Triton: `PTXAS error: Value 'sm_85' is not defined`**: set `ZLUDA_CC=8.6`
  and `TRITON_PTXAS_PATH` to ZLUDA's `ptxas.exe`.
- **Triton: undefined `__zluda_ptx_impl_ldmatrix_m8n8_x2_trans_b16`**: the
  `llvm_zluda` IR shim must be present (included in this tree).
- **Triton fails in `load_binary` with `Triton Error [CUDA]: error`**: check
  that `target\release\ptxas.exe` exists; rebuild it with
  `build-env.cmd cargo build --release -p ptxas`. Triton uses this stub to
  forward PTX to ZLUDA instead of producing NVIDIA SASS.
- **No clocks/power telemetry**: ZLUDA NVML reports `NotSupported` and the
  Windows ROCm 10.1 packages do not ship rocm-smi/amd-smi. Use the driver's
  fixed performance profile for reproducible runs; utilization can still be
  sampled with Windows GPU performance counters.
- **`cudaErrorSymbolNotFound` with PyTorch**: you are using a cu128+ wheel
  (SASS only). Use `cu118`.
- **`cublasLtMatmulAlgoGetHeuristic` / `cublasLtMatmulDescCreate` errors**:
  hipBLASLt has no gfx1010 kernels; the ZLUDA fallback covers the common paths.
  Optional workaround: `DISABLE_ADDMM_CUDA_LT=1`.
- **Stale or wrong Triton kernels**: delete `TRITON_CACHE_DIR` and
  `TORCHINDUCTOR_CACHE_DIR` contents.
- **`hipBLASLt` prints `rocblaslt error: ... TensileLibrary...`**: cosmetic;
  hipBLASLt reports missing kernels before the fallback handles the call.
- **OOM with free VRAM**: tune the caching allocator
  (`PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.9,max_split_size_mb:128`,
  `expandable_segments:True` where supported) and see Stage 6 in `ROADMAP.md`.

## 10. Repository layout

- `local-patches/zluda-rdna1-gfx1010.patch` — full patch against
  `v7-preview.21`.
- `RESULTS_RDNA1.md` — measurements and ZLUDA vs native ROCm comparison.
- `ROADMAP.md` — priorities, stages, acceptance metrics.
- `bench/`, `smoke/` — benchmark and validation scripts.
- `build-env.cmd`, `run-zluda.cmd` — build/run helpers.
