# CUDA graphs under ZLUDA on gfx1010

Status 2026-11-25: **working for PyTorch `torch.cuda.CUDAGraph`** after adding
three missing driver-API entry points to this fork.

## Why it matters for decode

The PyTorch decode loop is launch-bound on this stack: one token = ~24 layers x
~10 kernel launches. Measured decode throughput is 28-51 tok/s (ZLUDA/PyTorch)
versus 184-238 tok/s for llama.cpp on Vulkan (single command buffer per token),
so most of the gap is per-launch overhead, not GPU compute. CUDA graphs cut that
to one graph launch per token, which is exactly what llama.cpp benefits from.

## What was missing

PyTorch calls the CUDA runtime API; ZLUDA's runtime shim routes the old
(non-v2) entry points to the driver. Three of them were unimplemented, so
`torch.cuda.graph()` failed in `capture_begin` with `operation not supported`:

- `cuStreamBeginCapture` (legacy, no capture-mode argument)
- `cuStreamGetCaptureInfo` (legacy, 3-argument version)
- `cuGraphInstantiate` (without flags)

Everything else was already implemented (`cuStreamBeginCapture_v2`,
`cuStreamEndCapture`, `cuGraphInstantiateWithFlags`, `cuGraphLaunch`,
`cuGraphExecDestroy`, `cuGraphDestroy`, `cuStreamIsCapturing`).

## Implementation (this fork)

- `zluda/src/impl/stream.rs`
  - `begin_capture(stream)` -> `hipStreamBeginCapture(stream, GLOBAL)` (the
    legacy API has no mode argument; CUDA uses global capture semantics)
  - `get_capture_info(stream, status, id)` -> `hipStreamGetCaptureInfo`
- `zluda/src/impl/graph.rs`
  - `instantiate(exec, graph, error_node, log_buffer, buffer_size)` ->
    `hipGraphInstantiate`
- `zluda/src/lib.rs`: the three functions added to the `implemented` list.
- Debug aid: build-time gated logging of unimplemented API calls via
  `ZLUDA_DEBUG_UNIMPLEMENTED=1`, which is how the missing entry points were
  identified.

## Verified

Driver level (ctypes, side stream): `cuStreamBeginCapture_v2` -> 0,
`cuStreamEndCapture` -> 0, `cuGraphInstantiateWithFlags` -> 0,
`cuGraphLaunch` -> 0, results bit-exact.

PyTorch level (`torch.cuda.CUDAGraph` on a side stream):

```
capture ok
replay ok, max err 0.0
```

Limit: capturing the legacy default stream is unsupported by HIP (same
restriction as CUDA) - capture on a non-default (`torch.cuda.Stream`) stream,
as the PyTorch docs require.

## Next steps

- Re-run `torch.compile(mode="reduce-overhead")` with graphs enabled (Inductor
  cudagraph trees use side streams, so this is expected to work now).
- Validate `bench/graph_decode.py` (eager vs graph-replay decode) once the
  harness environment is stable; the measured kernel-launch count per eager
  decode step quantifies the remaining gap to Vulkan.
- bf16 emulation (`bf16` storage with fp32 compute) on ZLUDA: retest with
  in-place parameter conversion (the current implementation crashed without a
  report, most likely a VRAM spike on the 8 GB card).
## Decode gap analysis (why Vulkan is much faster at decode)

Measured on Qwen2.5-0.5B fp16, 1024-token context:

| quantity | value |
|---|---|
| PyTorch/ZLUDA eager decode | 34.0 ms/token (~29 tok/s) |
| llama.cpp/Vulkan decode | 4.2 ms/token (~240 tok/s) |
| trivial kernel launch on ZLUDA | 2.0 us |
| 240 launches/token (estimate) | ~0.5 ms |
| rocBLAS GEMV 1x896x896 | 24.7 us |
| rocBLAS GEMV 1x4864x896 (gate/up) | 47.5 us |
| rocBLAS GEMV 1x896x4864 (down_proj) | 251 us |
| Triton GEMV 1x896x896 | 21.6 us (1.14x faster) |
| Triton GEMV 1x4864x896 | 35.4 us (1.34x faster) |
| Triton GEMV 1x896x4864 | 35.8 us (7.0x faster) |

So kernel launches are cheap (2 us); CUDA graphs only remove ~0.5 ms/token and
cannot close an 8x gap alone. The decode gap comes from **the GEMM/GEMV kernels
themselves**: PyTorch's rocBLAS path spends 24.7-251 us per matvec (the MLP
down_proj is 251 us, 7x slower than an equivalent Triton kernel), while
llama.cpp uses hand-written bandwidth-bound GEMV kernels and packs a whole
token into a few command submissions. Attention (math SDPA with GQA expand) and
per-op elementwise/copies add the rest.

`zluda_ops.gemv` already provides fast Triton GEMV paths for small M and
`zluda_ops.auto` patches `F.linear` for decode shapes, but the end-to-end decode
improvement is not visible yet (26-29 tok/s measured with and without auto at
1k context), so the next step is to tune/extend those kernels for the exact
decode shapes (896/4864, bias, fp16) and fuse QKV/RoPE/KV-append rather than
relying on the generic path.

### torch.compile with cudagraphs: where it stops now

With the graph APIs implemented, `torch.compile(mode="reduce-overhead")` gets
through graph instantiation (`cuGraphInstantiateWithFlags` is called) and then
fails for two separate reasons:

1. Transformers' `DynamicCache` does `torch.cat` inside the compiled region -
   Inductor cudagraph trees reject writes to tensors captured from outside
   ("clone the tensor outside of torch.compile or use cudagraph_mark_step_begin").
   Fix on the model side: use `StaticCache` (fixed buffers).
2. Capturing the full forward with `StaticCache` then fails inside
   `cublasSgemm` during capture (`CUDA error: <unknown>`), i.e. ZLUDA's cuBLAS
   layer is not fully graph-capture-safe yet. Next step: audit `zluda_blas`
   for capture-unsafe behaviour (internal sync, stream handling) so that GEMMs
   can be recorded into a graph.

### bf16 emulation

bf16 emulation (bf16 weights, fp32 compute) works on native ROCm torch
(3812 tok/s prefill / 26.0 decode vs 2845/10.6 native bf16) but the ZLUDA
process died without a report (most likely VRAM: an fp32 copy of the model on
top of the bf16 one). Needs an in-place conversion path and a smaller-VRAM
recipe before it can be measured there.
