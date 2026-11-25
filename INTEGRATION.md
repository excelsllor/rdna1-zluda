# Out-of-the-box acceleration

`zluda_ops.auto` patches two PyTorch entry points at runtime:

- `torch.nn.functional.linear` → Triton GEMV for inference with M<=8, and
  bf16→fp32 emulation for small/medium bf16 weights.
- `torch.nn.functional.scaled_dot_product_attention` → causal Triton flash
  attention when the shape is supported.

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
mask regenerated in backward). Sliding window and varlen currently use the
correct math fallback (not the Triton kernel); Triton window/varlen kernels are
open. A 1-layer TransformerEncoder compiles in ~1 s; the full 8-layer
model is skipped because compile takes ~30 min even on a T4.

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
