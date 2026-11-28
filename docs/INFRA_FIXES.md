# Phase-1 infrastructure fixes log

## 2026-11-26: cuFFT and cuSPARSE now work

### cuFFT (cufft64_*.dll, crate zluda_fft)

Symptom: `torch.fft.*` failed with `CUFFT_NOT_SUPPORTED` (probe 2026-11-25).

Root cause found with the gated debug log (`ZLUDA_DEBUG_UNIMPLEMENTED=1`, added
to the cufft declarations macro): PyTorch's FFT path calls the generic
`cufftXtMakePlanMany` and `cufftXtExec`, which were unimplemented stubs
(returning NOT_SUPPORTED).

Fixes:
- `cufftXtMakePlanMany`: maps the CUDA `cudaDataType` input/output pair to the
  hipFFT transform type (R_32F/C_32F -> HIPFFT_R2C, C_32F/R_32F -> C2R,
  C_32F/C_32F -> C2C, f64 variants -> D2Z/Z2D/Z2Z), registers the transform kind
  per plan and calls `hipfftMakePlanMany64` (zluda_fft/src/impl.rs).
- `cufftXtExec`: looks up the registered transform kind and dispatches to the
  matching `hipfftExec*` (zluda_fft/src/impl.rs).
- Both are exported with a manual `#[no_mangle]` wrapper and registered via a
  `noop_extra` override so the macro does not also generate stubs
  (zluda_fft/src/lib.rs).

Verified: `torch.fft.fft`/`ifft` round-trip error 0.0, `rfft2`/`irfft2`
round-trip error 0.0, probe check `cufft_transforms` OK.

### cuSPARSE (cusparse64_*.dll, crate zluda_sparse)

Symptom: `torch.sparse.mm` failed at `cusparseSetStream` with
CUSPARSE_STATUS_NOT_SUPPORTED (probe 2026-11-25).

Fixes (iteratively found with `ZLUDA_DEBUG_UNIMPLEMENTED=1`):
- `cusparseSetStream`/`cusparseGetStream` mapped to
  `rocsparse_set_stream`/`rocsparse_get_stream` (plus the hip_runtime-sys
  dependency the crate lacked); the stream argument is converted with the
  existing `FromCuda` machinery.
- `cusparseXcoo2csr`: implemented via `rocsparse_coo2csr` (manual wrapper:
  cusparse index-base argument ignored, zero-base assumed as PyTorch always
  uses CUSPARSE_INDEX_BASE_ZERO for these paths).
- `cusparseCreateCsr`: implemented via `rocsparse_create_csr_descr` with explicit
  cudaDataType -> rocsparse_datatype mapping (f32/f32c/f64/f64c/i8) and the
  cusparse index-type numeric values reused as rocsparse index types (the enums
  share the same numbering: 16U=1, 32I=2, 64I=3).

Verified: `torch.sparse.mm` on a coalesced COO matrix matches the dense
reference exactly (max err 0.0), probe check `cusparse_spmm` OK.

### Remaining Phase-1 failures (with workarounds)

| check | status | workaround |
|---|---|---|
| cuDNN batchnorm | CUDNN_STATUS_NOT_SUPPORTED | `torch.backends.cudnn.enabled = False` -> native path works |
| cuDNN LSTM/GRU | CUDNN_STATUS_INTERNAL_ERROR + MIOpen kernel launch error | same native fallback works |
| fp64 GEMM | cublasDgemm returns NOT_SUPPORTED | no workaround yet (rocBLAS/Tensile f64 for gfx1010 missing); fp32/fp16/bf16 fine |

Next: implement MIOpen batchnorm/rnn mappings (zluda_dnn8), investigate the f64
GEMM path (rocBLAS Tensile coverage for gfx1010), then re-run
`bench/infra_probe.py`.
## 2026-11-26: cuDNN batchnorm and RNN fallbacks

cuDNN batchnorm and RNN/LSTM/GRU are not implemented in the `zluda_dnn8` mapping
(MIOpen bindings exist but there is no descriptor/RNG mapping yet). Two
different fallbacks are provided by `zluda_ops/cudnn_fallback.py`:

- **BatchNorm**: the `_BatchNorm.forward` is wrapped and runs with
  `torch.backends.cudnn.flags(enabled=False)` on ZLUDA (the native
  batch-norm path is correct; retrying after a failed cuDNN call is too late
  because the failed call can poison the MIOpen state). Installed by
  `install()`; `zluda_ops.auto` autoloads it unless `ZLUDA_CUDNN_FALLBACK=0`.
  Verified: `BatchNorm2d` forward OK.
- **LSTM/GRU/RNN**: the backend is chosen before the call, so wrapping the
  forward is not enough. Use `rnn_context()` around model creation AND forward:

      with zluda_ops.cudnn_fallback.rnn_context():
          model = torch.nn.LSTM(...).cuda()
          y = model(x)

  (equivalently `torch.backends.cudnn.enabled = False` for such models).
  Verified: LSTM and GRU forward OK inside the context.

Infra probe (`bench/infra_probe.py`) now covers both fallbacks:
`cudnn_batchnorm_fallback` and `cudnn_lstm_fallback` are OK; the raw cuDNN
calls remain FAIL (expected, documented above). Current result: **21/25**
checks pass; remaining failures: raw BN/RNN (use the fallbacks), fp64 GEMM
(rocBLAS/Tensile f64 not available for gfx1010).

## 2026-11-26: cublasSgemv (fp32 1-D GEMV) implemented

Found by the infra probe: `torch.mv` on fp32 tensors failed with
`CUBLAS_STATUS_NOT_SUPPORTED` (calling `cublasSgemv`). zluda_blas implemented
only GEMM variants; GEMV was never mapped.

Fix: added `sgemv`, `sgemv_v2`, `sgemv_v2_64` in `zluda_blas/src/impl.rs`
(mapping to `rocblas_sgemv` / `rocblas_sgemv_64`), registered the rocBLAS
symbols in the vtable and the cublas entry points in the implemented list.

Verified: `torch.mv` fp32 and fp16 now work (512-element output, fp16 path uses
the existing Hgemm? - `torch.mv` fp16 also OK), `w @ x` 1-D works.

Still unsupported: `cublasDgemv` (fp64 GEMV) - the crate has no `FromCuda`
conversion for `*const f64` arguments, and fp64 BLAS is generally unavailable on
this gfx1010 build (no f64 Tensile). fp64 `torch.mv`/`torch.linalg` remain
documented limitations.
