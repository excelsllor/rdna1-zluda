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