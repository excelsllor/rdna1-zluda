# Triton optimality audit and custom HIP/rocBLAS kernels (gfx1010)

Date: 2026-11-25. Goal: check whether the Triton kernels are optimal, and
whether writing custom HIP/rocBLAS kernels makes sense.

## 1. Triton kernels vs roofline

Hardware peaks (RX 5700 XT, gfx1010): FP32 9.75 TF, FP16 19.51 TF, memory
bandwidth 448 GB/s (paper). Measured achieved rates:

| kernel (shape) | time | achieved | % of peak | notes |
|---|---|---|---|---|
| GEMV 896x896 (qkv/o) | 23.5 us | 68 GB/s | 15% | launch/loop-bound, masked tail |
| GEMV 4864x896 (gate/up) | 48.2 us | 181 GB/s | 40% | memory-bound |
| GEMV 896x4864 (down) | 49.7 us | 176 GB/s | 39% | memory-bound |
| GEMV 151936x896 (lm_head) | 820 us | 332 GB/s | 74% | bandwidth-bound, good |
| FlashAttention prefill, S=2048, 14 heads, D=64 | 4.71 ms | 6.4 TF | 33% | solid for Triton FA |
| Flash-decoding, KV=8192, GQA 14/2 | 280.6 us | ~29 MB KV reads (7x redundancy) -> 104 GB/s | 23% | GQA reads KV per query head |
| fused RMSNorm / SwiGLU / GeGLU / RoPE / AdamW | - | bandwidth-bound, 1.5-10.9x faster than torch | near peak | already good |

Verdict: **Triton is not optimal for small matvecs (decode)**: 15-40% of peak
bandwidth on the per-layer GEMVs, and flash-decoding wastes 7x KV bandwidth
because the GQA factor is not exploited (query heads sharing KV heads). Fast
paths (lm_head, FA prefill, fused elementwise kernels) are near their practical
limits.

Triton-only improvements tried: a padded-K variant (BLOCK_K = next power of two,
mask-free single-layer reduce) gives only 1.2-1.4x (44.3 us qkv, 57.7 us
gate/up, 68.4 us down, 807 us lm_head) - still 2-4x behind the HIP prototype.
The row-per-program structure of Triton has inherent launch/ILP overhead.

## 2. Custom HIP kernels: feasibility and prototype

Feasibility verified end-to-end:

- compiler: ROCm 6.4 `clang++.exe` (LLVM 20) targets gfx1010 directly
  (`-x hip --offload-arch=gfx1010`), no CUDA needed; VS 2022 is the host
  toolchain (VS 2026 headers conflict with ROCm clang's HIP headers);
- a `__global__` HIP kernel plus an exported C launcher compiles into a DLL
  (`clang++ -shared -o gemv_fp16.dll ... -lamdhip64`);
- the DLL loads into the ZLUDA Python process via `ctypes` and launches on
  `hipDeviceSynchronize`-verified data; torch tensors are usable directly via
  `data_ptr()` (same HIP device memory).

Prototype: warp-per-row fp16 GEMV (`tools/hip_proto/gemv_fp16.cpp`), 4 warps per
block, half2 vectorized loads, warp shuffle reduction. Measured with warmups:

| shape (N x K) | HIP prototype | Triton (current) | speedup | HIP bandwidth |
|---|---|---|---|---|
| 896 x 896 (qkv/o) | 4.9 us | 23.5 us | **4.76x** | 325 GB/s |
| 4864 x 896 (gate/up) | 27.6 us | 48.2 us | **1.74x** | 316 GB/s |
| 896 x 4864 (down) | 29.9 us | 49.7 us | **1.66x** | 292 GB/s |
| 151936 x 896 (lm_head) | 751 us | 820 us | **1.09x** | 362 GB/s |

Correctness vs rocBLAS reference: max abs error 1e-5..2e-3 (fp16 reduction
order), which is at the fp16 noise level.

## 3. Is it worth it?

Kernel level: yes - the HIP GEMV reaches 65-81% of peak bandwidth instead of
15-74%, and the per-token estimate is ~135 us saved per layer (qkv 4x18.6 us +
gate/up 2x20.6 us + down 19.8 us) plus ~69 us on lm_head = **~3.3 ms/token of a
~11 ms graph-decode step (~30% faster decode, ~124 -> ~160 tok/s at 1k ctx)**.

Integration level: non-trivial, and this is the deciding factor:

1. **Streams**: ZLUDA maps CUDA streams to HIP streams, but launching from a
   raw HIP DLL on `hipStream 0` is *not ordered* with ZLUDA's streams - the
   first benchmark produced races (wrong results) and a hang; correctness
   required `hipDeviceSynchronize` inside the wrapper (which adds ~140 us per
   call and would defeat the gain). A proper solution needs the actual HIP
   stream handle that ZLUDA uses (unverified; passing torch's CUDA stream id
   hung the process) or a ZLUDA-side hook.
2. **CUDA graphs**: our graph-replayed decode captures the whole step; external
   HIP launches would need to be captured into the same HIP graph or run
   outside it. Unverified, and if they run outside the graph the launch savings
   disappear.
3. **Distribution**: a DLL per kernel family, ctypes dispatch, alignment/edge
   cases (K%2, N%ROW, fp16 reduce order) and no PyTorch custom-op machinery
   (the CUDA build of PyTorch cannot compile HIP extensions).
4. **Upstream/maintenance**: rocBLAS/Tensile already covers GEMM well (see
   below); a custom GEMV library duplicates a lot of work for one card family.

Recommendation:
- **Do not replace the full stack with custom HIP kernels.** Keep Triton for
  attention/fused kernels (it integrates with torch, autograd and graph
  capture).
- **Worth doing**: a small HIP (or better, Triton-plus-ZLUDA-hook) GEMV library
  for the 4 decode shapes, but only after a ZLUDA-side way to launch custom HIP
  kernels on the same HIP stream and inside CUDA-graph capture exists - then the
  estimated gain is ~30% decode. Next-cheapest alternatives in Triton:
  GQA KV-reuse in flash-decoding (up to ~2-3x on long-context decode
  attention), fusing QKV and gate+up GEMVs (fewer launches), and a multi-row
  per-program GEMV with vectorized loads.
- **Not worth it**: custom GEMM/Tensile replacements - rocBLAS 7.1.1 with the
  tuned gfx1010 Tensile overrides already reaches 8.18 TF of the 9.75 TF fp32
  peak (84%), and native ROCm torch (without overrides) is 4.6x slower.

## 4. Artifacts

- `tools/hip_proto/gemv_fp16.cpp`, `tools/hip_proto/build_gemv.cmd` - HIP GEMV
  prototype and build recipe (needs `build-env.cmd` with VS 2022 and ROCm
  clang).
- `bench/hip_gemv_proto.py` - benchmark harness if the DLL is built (loads via
  ctypes, compares with Triton and rocBLAS, checks correctness). Not wired into
  the model path; it exists to keep the measurement reproducible.