# Ecosystem roadmap: close infrastructure, then Triton kernels, then custom BLAS

Created 2026-11-25. This is the working plan for turning the ZLUDA/RDNA1 fork
from "benchmark demos work" into "everything that can work, works", then into a
kernel-optimised stack. Phases are ordered by dependency: nothing in phase 2/3
matters if the infrastructure underneath is flaky.

---

## Phase 1 - close the infrastructure and make everything possible work

Scope: applications, engines, libraries, functions. Exit criterion: a
documented status matrix per component with either "works + tested" (with a
reproducible test) or "documented blocker + fallback path", plus the top
blockers fixed.

### 1.1 Status matrix (audited 2026-11-25)

| component | status | evidence / notes | action |
|---|---|---|---|
| ZLUDA core (nvcuda.dll) | WORKS | PyTorch 2.7.1+cu118, llama.cpp, custom kernels; CUDA graphs now implemented (capture/instantiate/launch; launch-time attributes/PDL accepted) | keep; add regression tests |
| CUDA graphs | WORKS (basic) | torch.cuda.CUDAGraph capture/replay on side streams; self-loop update-in-graph variant broken | verify capture with cuBLAS in graph; document restrictions |
| cuBLAS -> rocBLAS (GEMM) | WORKS, near peak | 8.18 TF fp32 of 9.75 TF peak with tuned gfx1010 Tensile overrides; native ROCm torch without overrides is 4.6x slower | keep (phase 3 optional) |
| cuBLAS GEMV (small M) | WORKS but SLOW | 24-284 us per matvec; 15-40% of bandwidth; Triton/HIP GEMV prototypes 1.1-4.8x faster | phase 2 (Triton) then phase 3 |
| cuBLASLt (cublasLt64_11/12/13.dll) | MAPPED, UNVERIFIED | DLLs exist; PyTorch rarely exercises the epilogue paths | probe with a bias-epilogue GEMM / vendor code |
| cuDNN -> MIOpen: conv2d | WORKS (with solver selection) | fp16 3x3 picks GEMM solver; per-shape auto-selection implemented (fp32 Winograd cached-weight path 1.4-2.7x) | keep; extend to conv1d/3d/transposed |
| cuDNN: pool2d / avgpool / conv_transpose | WORKS | probe OK | regression tests |
| cuDNN: batchnorm | **WORKS via fallback (2026-11-26)** | cuDNN BN unmapped; `zluda_ops.cudnn_fallback` runs BN with cuDNN disabled (native path), auto-loaded by zluda_ops.auto | implement MIOpen BN mapping later |
| cuDNN: RNN (LSTM/GRU) | **FAIL/crash** | `CUDNN_STATUS_INTERNAL_ERROR` + MIOpen "Failed to launch kernel: invalid argument"; process died at exit | high priority: fix mapping or force native fallback |
| cuDNN: attention/FMHA, conv3d, RNN variants | UNTESTED | not exercised yet | probe matrix |
| cuFFT -> rocFFT (torch.fft) | **WORKS (fixed 2026-11-26)** | implemented `cufftXtMakePlanMany` + `cufftXtExec` (hipFFT type mapping); fft/ifft and rfft2/irfft2 round-trips exact | keep; add 3D/batched tests |
| cuSPARSE -> rocSPARSE | **WORKS (fixed 2026-11-26)** | implemented SetStream/GetStream, Xcoo2csr, CreateCsr (rocsparse mappings); torch.sparse COO mm exact | keep; add CSR/spmv/geam tests |
| cuRAND (philox) | WORKS (probe) | torch.randn OK; determinism/philox offset unverified | verify reproducibility |
| cuSOLVER (linalg.solve/svd/qr/cholesky) | NOT MAPPED | no cuSOLVER DLL in target/release | probe torch.linalg; add fallback to CPU or rocSOLVER mapping |
| NVML / NVAPI | MAPPED | nvml.dll/nvapi64.dll forwarding DLLs exist; on AMD NVML is absent (llama.cpp disables monitoring) | verify torch.cuda.memory stats + apps that require NVML |
| PyTorch dtypes: fp32 / fp16 / fp16-accum / bf16 / bf16-emul | WORKS | measured prefill/decode for all; fp16 accumulation is the best prefill knob; bf16-emul = 5488/29.6 tok/s | regression suite |
| SDPA backends | MATH only | flash/mem-efficient/cudnn SDPA unavailable; Triton FA + flex attention + flash-decoding provide the fast paths | document; keep dispatch |
| autograd / training | PARTIAL | Triton FA bwd works; conv fp32-cast gradients work; optimizer paths tested (8-bit AdamW). Full train-step matrix (AMP, gradcheck, quant-aware) not done | phase-1 test suite |
| flash_attn API shim | PARTIAL | prefill OK (5.1-5.5k tok/s); decode 1.6 tok/s anomaly | fix decode fallback |
| xformers API shim | PARTIAL | OOM in decode test; prefill path untested at scale | fix OOM |
| bitsandbytes | BLOCKED (excluded) | bnb kernels fail under ZLUDA; substitute: zluda_ops.optim8bit (4.6x faster step) | provide `bitsandbytes` API shim over optim8bit |
| llama.cpp (custom compute_86-PTX build) | WORKS (F16, fa=0) | full -ngl 99 offload; 6144/3294 prefill, 184 tok/s decode; fa=1 fails (mma.sync); Q4_K_M hangs (MMQ) | fix FA/Q4 or document fallbacks |
| llama.cpp official CUDA builds | BLOCKED | SASS-only, no PTX/sm_86 -> cannot run on ZLUDA | document (custom build is the path) |
| Ollama / Vulkan engine | WORKS (separate) | native Vulkan backend; used as a comparison baseline | keep as reference |
| ComfyUI (ZLUDA launchers) | LAUNCHERS EXIST, E2E UNTESTED | comfy-zluda.cmd / node launchers; conv auto solver ready | e2e SD1.5/SDXL/Flux smoke + perf |
| ComfyUI (native ROCm venv) | WORKS | D:\comfyui\Data\Packages\ComfyUIrocm (torch 2.14+rocm10.1) used for the ROCm baseline | keep as comparison |
| vLLM / TGI / ONNX Runtime CUDA EP | UNTESTED | vLLM needs paged-attention/triton ops; ONNX EP uses CUDA DLLs | probe feasibility, document |
| torch.compile | PARTIAL | prefill +42-49%; decode recompiles per token; cudagraph trees fail on DynamicCache and cuBLAS capture | phase 1: StaticCache + capture-safe path |
| OpenCL / D3D interop, video encode | UNTESTED | not needed for the current workloads | out of scope |

### 1.2 What must be verified for operability (Phase 1 checklist)

Run and record results for each item below (script per row, saved under
bench/results/infra/):

1. Library smoke matrix (per dtype fp32/fp16/bf16):
   fft 1D/2D/3D (C2C, R2C), sparse COO/CSR matmul + spmv + geam, cuSOLVER ops
   (solve/svd/qr/cholesky/eig), cuBLASLt epilogue GEMMs, cuDNN batchnorm (NCHW/
   NHWC), RNN LSTM/GRU, conv1d/conv3d/convTranspose, pooling 1d/2d/3d, softmax/
   layernorm variants, index/scatter/cumsum.
2. PyTorch training suite: forward+backward for Linear/Conv2d/RNN/LSTM/attention
   (SDPA math + Triton FA), AMP autocast fp16/bf16, gradient checkpointing,
   DDP-less single GPU.
3. Attention backends: math SDPA, SDPA fast (fallback), Triton FA, flex,
   flash_attn shim, xformers shim - correctness (vs math) + decode performance,
   under CUDA graphs.
4. Engines: llama.cpp (fa=0/1, F16/Q4/Q8), ComfyUI (SD1.5/SDXL/Flux single
   image), xformers/flash_attn shims end-to-end generation, a text-gen loop with
   GraphDecoder.
5. torch.compile: prefill (already measured), decode with StaticCache, cudagraph
   trees with the GEMV path.
6. Integration shims: bitsandbytes shim over optim8bit; flash_attn detect by
   transformers; xformers fallback path; ComfyUI nodes for ZLUDA conv/ops.
7. Tooling/CI: run-zluda regression script, patch check, watchdog safe runs,
   fixed-clock attempt, reproducible benchmark manifests.

### 1.3 What can still be completed / made to work (prioritised)

| # | item | effort | impact | notes |
|---|---|---|---|---|
| 1 | cuSPARSE stream mapping fix | S | makes torch.sparse usable | error is at cusparseSetStream; likely trivial mapping in zluda_sparse |
| 2 | cuFFT basic plan support | M | enables torch.fft (audio/video/diffusion) | map CUFFT plan types/sizes to hipFFT; needs tests for C2C/R2C |
| 3 | cuDNN batchnorm mapping | M | SD/ResNet inference paths | MIOpen batchnorm descriptors; or fallback to native torch batchnorm |
| 4 | RNN/LSTM crash fix | M | seq models, some audio | investigate descriptor/layout mapping; fallback to native at worst |
| 5 | cuSOLVER fallback | S-M | torch.linalg users | probe first; CPU fallback where possible |
| 6 | llama.cpp `-fa 1` on ZLUDA | L | llama.cpp with FA | mma.sync kernels; needs ZLUDA mma lowering or a non-mma FA path |
| 7 | llama.cpp Q4_K_M (MMQ) | M-L | quantized llama.cpp | debug MMQ hang (dp4a/wave ops?) |
| 8 | torch.compile decode | M | +1.5-2x decode | StaticCache + capture-safe path; cublas-in-graph audit |
| 9 | flash_attn shim decode | S | ecosystem compat | fallback math instead of slow path |
| 10 | xformers shim OOM | S | ecosystem compat | memory accounting in shim |
| 11 | bitsandbytes shim | S | drop-in for bnb users | wrap optim8bit (AdamW8bit) + stub int8 funcs |
| 12 | ComfyUI e2e | M | SD/Flux users | run single-image workflows, fix conv/attention gaps |
| 13 | int8 fused GEMV | S-M | quant decode | analogous to int4 path |
| 14 | profiler/CUPTI | S | observability | currently no kernel-level profiling; investigate or document |
| 15 | fixed-clock measurement harness | M | reproducible numbers | ADL unsupported; BIOS/power-play hack or accept variance |

### 1.4 Phase 1 acceptance criteria

- The status matrix above has an executable test for every WORKS row and a
  documented fallback for every FAIL row.
- No crash/hang in the smoke matrix (FFT/sparse/BN/RNN included); failures are
  clean errors with fallbacks.
- ComfyUI produces one image on ZLUDA end-to-end; llama.cpp produces text with
  the custom PTX build; a text-generation loop runs via GraphDecoder.
- Infra test suite runs in CI-like fashion (`tools/run_infra_suite.ps1`) and its
  results are committed under `bench/results/infra/`.

---

## Phase 2 - custom fast Triton kernels for everything

Scope: replace remaining slow kernels with tuned Triton implementations; keep
the dispatch/perf harness. Exit criterion: roofline table showing each kernel at
>= 60-70% of its practical limit, or documented reason why not.

### 2.1 Kernel inventory and targets

| kernel family | current | target | notes |
|---|---|---|---|
| GEMV (decode M=1..8) | 15-40% of BW | 60-80% | multi-row per program, vectorized half2 loads, pad K to power of two, fused QKV and gate+up |
| Flash-decoding (GQA) | 23% BW, 7x KV redundancy | 2-3x faster | KV-head-major programs sharing K/V across query heads, split-KV tuning |
| Flash-attention prefill | 33% of fp16 peak | 40-50% | already near Triton practical limit; try 2-CTA / persistent variants |
| Fused QKV + RoPE + KV append | 3 kernels | 1 kernel | decode-step fusion |
| MLP GEMV + activation | 2 kernels | 1 kernel | gate+up+SwiGLU fused for decode |
| lm_head GEMV | 332 GB/s | 380+ GB/s | 151936x896; split reduction over K |
| int4/int8 decode GEMV | fused int4 2.8x slower than dequant-once | fused at dequant-once speed | fused int8 missing entirely |
| int4 prefill GEMM | fused tiled OK | tune tiles for SD/LLM shapes | block tuning 64/128/256 |
| RMSNorm/SwiGLU/GeGLU/RoPE/softmax | near BW | >= 90% BW | already good; fold into decode fusion |
| AdamW 8-bit | good | keep | maybe fold grad clip |
| CE / sampling / top-k | torch | fused CE + sampling kernel | for generation loops |
| conv2d Triton | none (MIOpen) | evaluate | Winograd fp16 in Triton is hard; only if MIOpen path insufficient |
| bf16 emul kernels | uses fp32 GEMM | vectorized emul | optional |

### 2.2 Infrastructure for Phase 2

- `bench/roofline.py`: per-kernel achieved GB/s / TFLOPS vs peak, automatic.
- Autotune tables keyed by (arch, dtype, shape bucket) with a cache file; wire
  into `zluda_ops.auto` dispatch.
- Correctness suite for every kernel vs torch reference (fp16 tolerance),
  including backward paths.
- Graph-capture safety test for each kernel (must pass capture + replay).

---

## Phase 3 - replace hipBLAS/rocBLAS hot paths with custom kernels

Scope: only where phase 2 proves a gap that Triton cannot close inside the
ZLUDA/PyTorch constraints. Exit criterion: measured end-to-end gains on the
target workloads with fallbacks in place.

### 3.1 Candidates

| path | replace with | expected gain | risk |
|---|---|---|---|
| decode GEMV (4 shapes) | HIP warp-per-row kernels (prototype 1.1-4.8x) | ~30% decode | stream/graph integration in ZLUDA |
| quantized GEMV/GEMM | custom HIP (or Triton if it reaches target) | 1.5-2.5x | numerics/alignment |
| batched/strided BLAS calls | custom HIP | workload dependent | coverage |
| GEMM (large) | **do not replace** | - | rocBLAS+Tensile at 84% of peak already |
| BLASLt epilogue fusions | custom HIP | unknown | PyTorch usage unclear |

### 3.2 Integration architecture (prerequisite)

- Add a ZLUDA-side facility to launch custom HIP kernels on the same HIP stream
  that ZLUDA uses for CUDA stream objects, and to be visible to CUDA-graph
  capture (either ZLUDA translates the captured launch into a HIP graph node, or
  expose a small C API that our DLL can call with the ZLUDA stream handle).
- Fallback: if a custom kernel cannot be captured, run it outside the graph and
  account for the extra sync.
- Distribution: a single `zluda_gemv.dll` shipped alongside the fork, loaded
  lazily via ctypes, with a pure-Triton fallback when missing.

### 3.3 Milestones

1. ZLUDA stream/graph integration for external HIP kernels (POC: elementwise).
2. GEMV library for the 4 decode shapes, correctness + graph replay.
3. Integration into `zluda_ops.auto` dispatch with env flag + fallback.
4. End-to-end measurement (GraphDecoder, ComfyUI, llama.cpp compare) and
   decision on quantized kernels.

---

## Immediate next actions (Phase 1 kickoff)

1. `tools/run_infra_suite.ps1` + `bench/infra_probe.py`: implement the smoke
   matrix from 1.2 (library ops, dtypes), record results.
2. Fix in order: cuSPARSE stream (S), cuFFT basic (M), cuDNN batchnorm (M),
   RNN/LSTM (M); each with a clean fallback and a regression test.
3. torch.compile decode with StaticCache + GEMV path; llama.cpp FA/Q4
   investigation; shims (flash_attn decode, xformers OOM); bnb shim.
4. ComfyUI end-to-end smoke (SD1.5/SDXL/Flux) with the conv auto-solver.
5. Commit results to `bench/results/infra/` and update this document's matrix.
