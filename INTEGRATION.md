# Out-of-the-box acceleration

`zluda_ops.auto` patches two PyTorch entry points at runtime:

- `torch.nn.functional.linear` → Triton GEMV for inference with M<=8, and
  bf16→fp32 emulation for small/medium bf16 weights.
- `torch.nn.functional.scaled_dot_product_attention` → causal Triton flash
  attention when the shape is supported, and decode-shaped calls
  (M<=16 query tokens, KV>=512, fp16, no mask/dropout/grad) to flash-decoding
  (split-KV).

Everything else falls back to the original PyTorch function. The patch is
Python-level; code that caches `F.linear`/SDPA before installation, or runs
inside a compiled graph, bypasses it.

## Use

Scripts (venv):

```bat
run-zluda-auto.cmd venv\Scripts\python.exe app.py
```

or explicitly:

```python
import zluda_ops.auto
```

Training / MHA-heavy models (P0 workaround for the cuDNN FMHA crash):

```bat
run-zluda-train.cmd path\to\train.py [args...]
```

This mode:
- installs a slow-path `nn.MultiheadAttention` that avoids the unsupported
  cuDNN/CUTLASS FMHA kernel and excludes it from Dynamo tracing;
- disables the inference auto-dispatch (`ZLUDA_AUTO_OPS=0`) so `torch.compile`
  is not disturbed by the `F.linear` wrapper;
- caps VRAM at 85% (`ZLUDA_VRAM_FRACTION`).
- `torch.compile` for this model is intentionally skipped: it takes ~30
  minutes even on an NVIDIA T4 and is not a P0 blocker.

ComfyUI: `tools\install_auto.py` installs the `zluda_ops_auto` custom node,
which applies the patch after ComfyUI has configured the allocator:

```bat
venv\Scripts\python.exe tools\install_auto.py
```

Use the custom-nodes launcher (`comfy-zluda-nodes.cmd`). The base
`--disable-all-custom-nodes` mode does not load the patch. On startup the log
contains `zluda_ops.auto: patched F.linear and F.scaled_dot_product_attention`.

Controls:

| Variable | Default | Effect |
|---|---|---|
| `ZLUDA_AUTO_OPS=1` | off | load the inference patch through the venv `.pth` |
| `ZLUDA_OPS_DISABLE=1` | off | never install the patch |
| `ZLUDA_AUTO_VERBOSE=1` | off | print the install message |
| `ZLUDA_AUTO_BF16_MAX_MB` | 64 | max bf16 weight size for auto emulation; 0 disables |
| `ZLUDA_MHA_SLOW=1` | off | force the slow math path for `nn.MultiheadAttention` |
| `ZLUDA_FP16_ACC=1` | off | experimental HH fp16 accumulation for training |
| ZLUDA_CONV_FP32=1 | off | run eligible fp16 3x3 convs through the fp32 Winograd path (1.4-4.3x forward); needs import zluda_ops.auto |
| `ZLUDA_VRAM_FRACTION` | 0.85 | per-process VRAM fraction for `run-zluda-train.cmd` |

Safety conditions:

- GEMV: CUDA, 2D contiguous activation/weight, fp16/fp32, M=1..8, no
  requires-grad tensors (inference only), arbitrary K (masked tail).
- bf16: CUDA, 2D contiguous, weights up to the configured memory limit.
- SDPA: causal, fp16, head_dim=64, sequence length divisible by 128, no mask,
  no dropout, no compiled graph.
- Any exception or unsupported case calls the original PyTorch function.

## Measured effect of the auto path

| Case | Original PyTorch | Auto | Speedup | Max abs err |
|---|---|---|---|---|
| linear fp16 M=1, K=4096 | 0.292 ms | 0.150 ms | 1.95x | 0.00024 |
| linear fp16 M=4, K=4096 | 0.309 ms | 0.150 ms | 2.06x | 0.00024 |
| linear fp16 M=8, K=4096 | 0.289 ms | 0.199 ms | 1.45x | 0.00049 |
| linear fp32 M=1, K=4096 | 0.459 ms | 0.219 ms | 2.10x | 0 |
| linear fp32 M=8, K=4096 | 0.447 ms | 0.365 ms | 1.22x | 0 |
| linear fp16 M=1, K=3200 | 0.193 ms | 0.118 ms | 1.63x | 0.00024 |
| bf16 linear 64x1024x1024 | 0.272 ms | 0.165 ms | 1.65x | 0.002 |
| bf16 linear 64x4096x4096 | 2.227 ms | 1.243 ms | 1.79x | 0.016 |
| causal SDPA 512, fp16, D64 | 0.657 ms | 0.413 ms | 1.59x | 0.001 |

The auto GEMV path uses full fp32 products and fp32 accumulation (slightly
slower than the `bench/gemv.py` fast mode, which uses fp16 products for the
reported peak numbers). bf16 auto uses fp32 conversion, fp32 GEMM and casts
back.

## fp16 accumulation

`zluda_ops.fp16acc.fp16_accumulation(enable=True)` is an opt-in context manager
that warns and restores the previous setting. Use it only for inference and
validate accuracy (`bench/fp16_ux.py`); it is off by default.

## Attention API

Causal and non-causal Triton FA work for fp16, head_dim=64, sequence lengths
divisible by 128 and no dropout. Boolean and additive attention masks (2D, 3D
or 4D with batch/head broadcast, optionally combined with causal) run in the
Triton kernel for forward and backward; masks that are not broadcastable fall
back to math SDPA. `flash_attn_func` supports sliding
`window_size`; `flash_attn_varlen_func` accepts packed q/k/v and
`cu_seqlens`. Dropout also runs in the Triton kernels (inverted dropout, same
mask regenerated in backward), and the sliding window is arithmetic in the
kernel (no dense mask). Varlen dispatches per sequence through Triton; a fused
packed varlen kernel is open. A 1-layer TransformerEncoder compiles in ~1 s;
the full 8-layer model is skipped because compile takes ~30 min even on a T4.

## Training with HH (experimental)

`run-zluda-train-hh.cmd path\to\train.py` enables global fp16 accumulation
(`ZLUDA_FP16_ACC=1`) for training. The `gfx1010_hh.csv` override will apply to
matching HH GEMMs. This is not the default: fp16 accumulation can reduce
gradient accuracy and should be validated for your model. `run-zluda-train.cmd`
keeps HPA (fp32 accumulation).

## Accumulation policy

| Path | Inputs | Products | Accumulation | Output |
|---|---|---|---|---|
| rocBLAS GEMM fp32 | fp32 | fp32 | **fp32** | fp32 |
| rocBLAS GEMM fp16 (default) | fp16 | fp16 multiply | **fp32** | fp16 |
| rocBLAS HH (opt-in) | fp16 | packed fp16 | fp16 | fp16 |
| Triton GEMV auto | fp16/fp32 | **fp32** | **fp32** | input dtype |
| Triton GEMV bench fast | fp16 | fp16 | fp32 | fp16 |
| bf16 emulation | bf16→fp32 | fp32 | **fp32** | bf16 |
| Triton causal FA | fp16 | fp16 matmul | **fp32 softmax/LSE/accum** | fp16 |

`int16` is not used anywhere. `int8` GEMM is not available through ZLUDA
(`CUBLAS_STATUS_INVALID_VALUE`), so there is no int8 accumulation path.
fp16 accumulation for rocBLAS is opt-in only via
`torch._C._set_cublas_allow_fp16_accumulation(True)`; the auto layer never
enables it.

## Next steps

1. Validate the auto path on real ComfyUI SD/Flux/GGUF workflows and
   llama.cpp-style decode; record tokens/s and image timings.
2. Extend GEMV to M=16/32 (split-K, packed-half reduction, dwordx4 unpacking)
   and add an inference-safe autograd-free fallback contract.
3. Dispatch more operators where profitable (`matmul` for skinny bf16,
   `addmm`, fused bias/residual) while preserving the fallback rules.
4. Finish non-causal Triton FA backward, sliding window, varlen, softcap and
   dropout; add flash-decoding for long-context inference.
5. Re-tune Tensile for fp32 8.8–9.2 TF and HH 14–16 TF; evaluate int8
   quantization (Stage 3).
6. Keep `ZLUDA_OPS_DISABLE=1` as the rollback switch for any regression.

## Ecosystem shims: flash_attn and xformers (no CUDA build required)

Real `flash-attn` and `xformers` wheels cannot be used on this stack (they need
nvcc/CUTLASS sm80+ kernels, and RDNA1 has no matrix cores). The repo ships API
shims backed by the Triton kernels instead:

```bat
venv\Scripts\python.exe tools\install_ecosystem.py
venv\Scripts\python.exe -c "import flash_attn, xformers; print(flash_attn.__version__, xformers.__version__)"
```

- `integrations/flash_attn`: `flash_attn_func`, `flash_attn_varlen_func`,
  `FlashAttnFunc`, `flash_attn.bert_padding.unpad_input/pad_input`, version
  string. Layout is (B, S, H, D) like the real package. The Triton FA /
  flash-decoding path is used when supported (fp16, head_dim 64, causal,
  sequence divisible by 128); otherwise it falls back to math SDPA.
- `integrations/xformers`: `xformers.ops.memory_efficient_attention`,
  `memory_efficient_attention_forward`, `AttentionBias`,
  `LowerTriangularMask`, `TensorBias`, `BlockDiagonalMask`, `fmha` aliases.
  (B, S, H, D) tensors, additive biases and causal markers are mapped to the
  Triton path with a math-SDPA fallback.
- The installer writes `zluda_ecosystem.pth` into the venv site-packages (adds
  the repo root and `integrations/` to sys.path). Remove it or run the installer
  with `--uninstall` to roll back.

Limitations: no real CUDA flash-attention kernels (no matrix cores on RDNA1),
no xformers CUTLASS kernels, no block-sparse/attention-bias kernels beyond the
math fallback. `transformers` can detect the shim via `is_flash_attn_2_available`
and will exercise the Triton path for supported shapes.

### llama.cpp on ZLUDA (blocked - do not run with GPU offload)

**SUPERSEDED (2026-11-26): a custom compute_86-PTX build runs llama.cpp with
-ngl 99` (F16, `-fa 0`); see the next section.**

With the launch-attribute patch (`zluda/src/impl/driver.rs`) llama.cpp's CUDA
backend loads under ZLUDA and runs at `-ngl 0` (CPU: pp16 124 tok/s, tg2 17.4)
and `-ngl 1 -fa 0` with the F16 GGUF (pp8 415.6, tg1 61.5). Increasing the
offloaded layers causes launch failures or **hard AMD driver crashes that
require a reboot**. The cause is ZLUDA's execution/JIT of llama.cpp's CUDA
kernel set, not the launch API; a same-engine Vulkan-vs-ZLUDA comparison stays
blocked. Details and the ready matrix runner: ENGINE_COMPARISON.md and
bench/llamacpp_zluda_matrix.ps1.

### llama.cpp on ZLUDA: now works (F16, no flash attention)

A local llama.cpp build with compute_86 PTX (`86-virtual`, CUDA 12.4.1 portable
toolchain, VS 2022) runs under ZLUDA with full GPU offload (`-ngl 99`).
Same-engine Vulkan-vs-ZLUDA numbers (Qwen2.5-0.5B F16, FA=0): ZLUDA 6144/3294
tok/s prefill at 2k/8k and 184 tok/s decode vs Vulkan 6258/4179 and 212/173.
llama.cpp FA (`mma.sync`) and Q4_K_M (MMQ) kernels are still unsupported by
ZLUDA on RDNA1. See docs/LLAMACPP_VULKAN_VS_ZLUDA.md and
docs/LLAMACPP_ZLUDA_BUILD.md; run risky tests through tools/gpu_watchdog.ps1.
