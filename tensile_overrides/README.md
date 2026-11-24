# gfx1010 HH Tensile override

`rocBLAS 7.1.1` in this runtime already contains **110 HH solutions** for
gfx1010. The default selection is not the fastest. This override chooses
existing HH kernels with a better balance for square 1024/2048/4096 shapes.

| n | default HH | override HH | speedup |
|---|---|---|---|
| 1024 | ~12.4 TF | ~12.9 TF | 1.04x |
| 2048 | ~9.5 TF | **17.94 TF** | 1.89x |
| 4096 | ~12.1 TF | **16.73 TF** | 1.38x |

Measured with random fp16 inputs and
`torch._C._set_cublas_allow_fp16_accumulation(True)`. `run-zluda.cmd` sets
`ROCBLAS_TENSILE_GEMM_OVERRIDE_PATH` and `ROCBLAS_USE_HIPBLASLT=0`
automatically when this file exists.

## Format

The file is consumed by rocBLAS:

```text
transA,transB,M,N,batch_count,K,alpha,beta,lda,ldb,ldc,input_type,output_type,compute_type,solution_index
```

The raw indices were derived from `rocblas_gemm_ex_get_solutions` using
`raw = -solution_index - 10` on this exact rocBLAS 7.1.1 build. They are
**not portable** to another rocBLAS release or device architecture: indices
are build-local. Remove the file or unset the environment variable to roll
back.

## Important

- This profile applies only to HH (fp16 in/out, fp16 compute); it does not
  change HPA (fp32 accumulation) or SS (fp32) unless corresponding lines are
  added.
- Random data is required for realistic measurements. Zeros, ones and
  uninitialized buffers are compressed/boost differently and gave ~9.1 TF
  for SS versus ~8.1–8.2 TF with random data.
- The main remaining P0-5 gap is SS (~8.2 TF) and HPA (~7.2 TF), not the
  availability of HH kernels.
