use cuda_types::{cublas::*, cublaslt::*};
use hip_runtime_sys::hipStream_t;
use hipblaslt_sys::*;
use std::collections::HashMap;
use std::mem;
use std::sync::{Mutex, OnceLock};
use zluda_common::{constants, FromCuda, ZludaObject};

// Marker written into a synthetic cublasLtMatmulAlgo_t to request the
// rocBLAS fallback (used when hipBLASLt has no kernels for this GPU).
const FALLBACK_ALGO_MAGIC: u64 = 0x5A4C_5544_415F_4C54;

static HIPBLASLT_UNAVAILABLE: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

#[derive(Clone, Copy)]
struct LayoutInfo {
    dtype: i32,
    rows: u64,
    cols: u64,
    ld: i64,
    order: i32,
    batch_count: i32,
}

#[derive(Clone, Copy)]
struct DescInfo {
    compute_type: i32,
    scale_type: i32,
    trans_a: i32,
    trans_b: i32,
    epilogue: i32,
    bias_ptr: usize,
    bias_dtype: i32,
    force_rocblas: bool,
}

fn layouts() -> &'static Mutex<HashMap<usize, LayoutInfo>> {
    static M: OnceLock<Mutex<HashMap<usize, LayoutInfo>>> = OnceLock::new();
    M.get_or_init(|| Mutex::new(HashMap::new()))
}

fn descs() -> &'static Mutex<HashMap<usize, DescInfo>> {
    static M: OnceLock<Mutex<HashMap<usize, DescInfo>>> = OnceLock::new();
    M.get_or_init(|| Mutex::new(HashMap::new()))
}

#[cfg(windows)]
fn rocm_dtype(hip_dtype: i32) -> Option<i32> {
    use crate::rocblas::*;
    let dt = if hip_dtype == hipDataType::HIP_R_16F.0 as i32 {
        DT_F16
    } else if hip_dtype == hipDataType::HIP_R_32F.0 as i32 {
        DT_F32
    } else if hip_dtype == hipDataType::HIP_R_64F.0 as i32 {
        DT_F64
    } else if hip_dtype == hipDataType::HIP_R_16BF.0 as i32 {
        DT_BF16
    } else if hip_dtype == hipDataType::HIP_R_8I.0 as i32 {
        DT_I8
    } else {
        return None;
    };
    Some(dt)
}

#[cfg(windows)]
fn rocm_compute(hip_compute: i32) -> Option<i32> {
    use crate::rocblas::*;
    let ct = if hip_compute == hipblasComputeType_t::HIPBLAS_COMPUTE_16F.0 as i32 {
        DT_F16
    } else if hip_compute == hipblasComputeType_t::HIPBLAS_COMPUTE_32F.0 as i32
        || hip_compute == hipblasComputeType_t::HIPBLAS_COMPUTE_32F_PEDANTIC.0 as i32
        || hip_compute == hipblasComputeType_t::HIPBLAS_COMPUTE_32F_FAST_16F.0 as i32
        || hip_compute == hipblasComputeType_t::HIPBLAS_COMPUTE_32F_FAST_16BF.0 as i32
        || hip_compute == hipblasComputeType_t::HIPBLAS_COMPUTE_32F_FAST_TF32.0 as i32
    {
        DT_F32
    } else if hip_compute == hipblasComputeType_t::HIPBLAS_COMPUTE_64F.0 as i32 {
        DT_F64
    } else if hip_compute == hipblasComputeType_t::HIPBLAS_COMPUTE_32I.0 as i32 {
        DT_I32
    } else {
        return None;
    };
    Some(ct)
}

#[cfg(windows)]
fn rocm_op(hip_op: i32) -> Option<i32> {
    use crate::rocblas::*;
    if hip_op == hipblasOperation_t::HIPBLAS_OP_N.0 as i32 {
        Some(OP_N)
    } else if hip_op == hipblasOperation_t::HIPBLAS_OP_T.0 as i32 {
        Some(OP_T)
    } else {
        None
    }
}

fn hipblaslt() -> Result<&'static super::HipblasltVtable, hipblasLtError> {
    static LOCK: OnceLock<Result<super::HipblasltVtable, hipblasLtError>> = OnceLock::new();
    let unwrapped: &Result<super::HipblasltVtable, hipblasLtError> =
        LOCK.get_or_init(|| unsafe { super::HipblasltVtable::new() });
    unwrapped.as_ref().map_err(|x| *x)
}

pub struct Handle {
    handle: hipblasLtHandle_t,
}

impl Handle {
    fn new(handle: hipblasLtHandle_t) -> Self {
        Self { handle }
    }
}

impl ZludaObject for Handle {
    const COOKIE: usize = 0xd1d9cb43416c9620;
    type Error = cublasError_t;
    type CudaHandle = cublasLtHandle_t;
    fn drop_checked(&mut self) -> cublasStatus_t {
        unsafe { hipblaslt()?.hipblasLtDestroy(self.handle) }?;
        Ok(())
    }
}

#[repr(transparent)]
struct WrappedCublasLtHandle(cublasLtHandle_t);

unsafe impl Send for WrappedCublasLtHandle {}
unsafe impl Sync for WrappedCublasLtHandle {}

// This is repr(c) because it needs to be compatible with Handle in zluda_blas.
#[repr(C)]
struct BlasHandle {
    blas_lt: WrappedCublasLtHandle,
}

impl ZludaObject for BlasHandle {
    const COOKIE: usize = constants::BLAS_HANDLE_COOKIE;
    type Error = cublasError_t;
    type CudaHandle = cublasLtHandle_t;
    fn drop_checked(&mut self) -> cublasStatus_t {
        // This is managed by zluda_blas
        Ok(())
    }
}

#[allow(unused_imports)]
impl<'a> FromCuda<'a, cublasLtHandle_t, cublasError_t> for &'a Handle {
    fn from_cuda(handle: &'a cublasLtHandle_t) -> Result<&'a Handle, cublasError_t> {
        // Try interpreting as a direct BlasLtHandle
        let live_check = zluda_common::as_ref(handle).ok_or(cublasError_t::INVALID_VALUE)?;
        if let Ok(blas_lt) = live_check.as_result() {
            return Ok(blas_lt);
        }

        // Fallback: handle may be a BlasHandle with an embedded BlasLtHandle
        let live_check =
            zluda_common::as_ref::<BlasHandle>(handle).ok_or(cublasError_t::INVALID_VALUE)?;
        let blas_handle = live_check.as_result()?;
        Ok(zluda_common::as_ref::<Handle>(&blas_handle.blas_lt.0)
            .ok_or(cublasError_t::INVALID_VALUE)?
            .as_result()?)
    }
}
const _: fn() = || {
    let _ = std::mem::transmute::<<Handle as zluda_common::ZludaObject>::CudaHandle, usize>;
};

#[cfg(debug_assertions)]
pub(crate) fn unimplemented() -> cublasStatus_t {
    unimplemented!()
}

#[cfg(not(debug_assertions))]
pub(crate) fn unimplemented() -> cublasStatus_t {
    cublasStatus_t::ERROR_NOT_SUPPORTED
}

// Same as in zluda_blas
pub(crate) fn get_status_name(status: cublasStatus_t) -> *const ::core::ffi::c_char {
    match status {
        cublasStatus_t::SUCCESS => c"CUBLAS_STATUS_SUCCESS".as_ptr(),
        cublasStatus_t::ERROR_NOT_INITIALIZED => c"CUBLAS_STATUS_NOT_INITIALIZED".as_ptr(),
        cublasStatus_t::ERROR_ALLOC_FAILED => c"CUBLAS_STATUS_ALLOC_FAILED".as_ptr(),
        cublasStatus_t::ERROR_INVALID_VALUE => c"CUBLAS_STATUS_INVALID_VALUE".as_ptr(),
        cublasStatus_t::ERROR_ARCH_MISMATCH => c"CUBLAS_STATUS_ARCH_MISMATCH".as_ptr(),
        cublasStatus_t::ERROR_MAPPING_ERROR => c"CUBLAS_STATUS_MAPPING_ERROR".as_ptr(),
        cublasStatus_t::ERROR_EXECUTION_FAILED => c"CUBLAS_STATUS_EXECUTION_FAILED".as_ptr(),
        cublasStatus_t::ERROR_INTERNAL_ERROR => c"CUBLAS_STATUS_INTERNAL_ERROR".as_ptr(),
        cublasStatus_t::ERROR_NOT_SUPPORTED => c"CUBLAS_STATUS_NOT_SUPPORTED".as_ptr(),
        cublasStatus_t::ERROR_LICENSE_ERROR => c"CUBLAS_STATUS_LICENSE_ERROR".as_ptr(),
        _ => c"CUBLAS_STATUS_UNKNOWN".as_ptr(),
    }
}

pub(crate) fn get_status_string(status: cublasStatus_t) -> *const ::core::ffi::c_char {
    if cfg!(windows) && status.is_err() && status.err() == hipblaslt().err().map(Into::into) {
        return c"Neither hipblaslt.dll nor libhipblaslt.dll could be found. Please install HIP SDK: https://zluda.readthedocs.io/latest/hip_sdk.html".as_ptr();
    }
    get_status_name(status)
}

pub(crate) fn get_version() -> usize {
    todo!()
}

pub(crate) fn get_cudart_version() -> usize {
    todo!()
}

pub(crate) fn disable_cpu_instructions_set_mask(_mask: ::core::ffi::c_uint) -> ::core::ffi::c_uint {
    todo!()
}

pub(crate) fn create(handle: &mut cublasLtHandle_t) -> cublasStatus_t {
    let mut hipblas_lt = unsafe { std::mem::zeroed() };
    unsafe { hipblaslt()?.hipblasLtCreate(&mut hipblas_lt) }?;

    let zluda_handle = Handle::new(hipblas_lt);

    *handle = Handle::wrap(zluda_handle);
    Ok(())
}

pub(crate) fn destroy(handle: cublasLtHandle_t) -> cublasStatus_t {
    // Try interpreting as a direct BlasLtHandle
    let result = zluda_common::drop_checked::<Handle>(handle);
    if result != cublasStatus_t::ERROR_INVALID_VALUE {
        return result;
    }

    // Fallback: handle may be a BlasHandle with an embedded blas_lt
    let blas_handle = zluda_common::as_ref::<BlasHandle>(&handle)
        .ok_or(cublasError_t::INVALID_VALUE)?
        .as_result()?;
    zluda_common::drop_checked::<Handle>(blas_handle.blas_lt.0)
}

fn cuda_algo_from_hip(hip: hipblasLtMatmulAlgo_t) -> cublasLtMatmulAlgo_t {
    let mut cuda = cublasLtMatmulAlgo_t { data: [0; 8] };
    let (chunks, _) = hip.data.as_chunks::<8>();
    cuda.data[0] = u64::from_ne_bytes(chunks[0]);
    cuda.data[1] = u64::from_ne_bytes(chunks[1]);
    cuda.data[2] = hip.max_workspace_bytes as u64;
    cuda
}

pub(crate) fn matmul(
    light_handle: &Handle,
    compute_desc: hipblasLtMatmulDesc_t,
    alpha: *const ::core::ffi::c_void,
    a: *const ::core::ffi::c_void,
    a_desc: hipblasLtMatrixLayout_t,
    b: *const ::core::ffi::c_void,
    b_desc: hipblasLtMatrixLayout_t,
    beta: *const ::core::ffi::c_void,
    c: *const ::core::ffi::c_void,
    c_desc: hipblasLtMatrixLayout_t,
    d: *mut ::core::ffi::c_void,
    d_desc: hipblasLtMatrixLayout_t,
    algo: hipblasLtMatmulAlgo_t,
    workspace: *mut ::core::ffi::c_void,
    workspace_size_in_bytes: usize,
    stream: hipStream_t,
) -> cublasStatus_t {
    let mut algo_prefix = [0u8; 8];
    algo_prefix.copy_from_slice(&algo.data[..8]);
    let fallback_requested = u64::from_ne_bytes(algo_prefix) == FALLBACK_ALGO_MAGIC;

    if !fallback_requested
        && !HIPBLASLT_UNAVAILABLE.load(std::sync::atomic::Ordering::Relaxed)
    {
        let result = unsafe {
            hipblaslt()?.hipblasLtMatmul(
                light_handle.handle,
                compute_desc,
                alpha,
                a,
                a_desc,
                b,
                b_desc,
                beta,
                c,
                c_desc,
                d,
                d_desc,
                &algo,
                workspace,
                workspace_size_in_bytes,
                stream,
            )
        };
        if result.is_ok() {
            return Ok(());
        }
        HIPBLASLT_UNAVAILABLE.store(true, std::sync::atomic::Ordering::Relaxed);
    }

    #[cfg(windows)]
    {
        if unsafe {
            rocblas_fallback(
                compute_desc, alpha, a, a_desc, b, b_desc, beta, c, c_desc, d, d_desc, stream,
            )
        }
        .is_ok()
        {
            return Ok(());
        }
    }

    cublasStatus_t::ERROR_NOT_SUPPORTED
}

#[cfg(windows)]
unsafe fn rocblas_fallback(
    compute_desc: hipblasLtMatmulDesc_t,
    alpha: *const ::core::ffi::c_void,
    a: *const ::core::ffi::c_void,
    a_desc: hipblasLtMatrixLayout_t,
    b: *const ::core::ffi::c_void,
    b_desc: hipblasLtMatrixLayout_t,
    beta: *const ::core::ffi::c_void,
    c: *const ::core::ffi::c_void,
    c_desc: hipblasLtMatrixLayout_t,
    d: *mut ::core::ffi::c_void,
    d_desc: hipblasLtMatrixLayout_t,
    stream: hipStream_t,
) -> Result<(), ()> {
    let desc = descs()
        .lock()
        .unwrap()
        .get(&(compute_desc.0 as usize))
        .copied()
        .ok_or(())?;
    let a_info = layouts()
        .lock()
        .unwrap()
        .get(&(a_desc.0 as usize))
        .copied()
        .ok_or(())?;
    let b_info = layouts()
        .lock()
        .unwrap()
        .get(&(b_desc.0 as usize))
        .copied()
        .ok_or(())?;
    let c_info = layouts()
        .lock()
        .unwrap()
        .get(&(c_desc.0 as usize))
        .copied()
        .ok_or(())?;
    let d_info = layouts()
        .lock()
        .unwrap()
        .get(&(d_desc.0 as usize))
        .copied()
        .ok_or(())?;

    // rocBLAS fallback only supports plain column-major, single-batch GEMMs.
    for info in [a_info, b_info, c_info, d_info] {
        if info.order != 0 || info.batch_count > 1 {
            return Err(());
        }
    }

    let a_type = rocm_dtype(a_info.dtype).ok_or(())?;
    let b_type = rocm_dtype(b_info.dtype).ok_or(())?;
    let c_type = rocm_dtype(c_info.dtype).ok_or(())?;
    let d_type = rocm_dtype(d_info.dtype).ok_or(())?;
    let compute_type = rocm_compute(desc.compute_type).ok_or(())?;
    let ta = rocm_op(desc.trans_a).ok_or(())?;
    let tb = rocm_op(desc.trans_b).ok_or(())?;

    let m = i32::try_from(d_info.rows).map_err(|_| ())?;
    let n = i32::try_from(d_info.cols).map_err(|_| ())?;
    let k = if ta == crate::rocblas::OP_N {
        i32::try_from(a_info.cols).map_err(|_| ())?
    } else {
        i32::try_from(a_info.rows).map_err(|_| ())?
    };
    if m <= 0 || n <= 0 || k < 0 {
        return Err(());
    }
    if a.is_null() || b.is_null() || d.is_null() {
        return Err(());
    }

    let lda = i32::try_from(a_info.ld).map_err(|_| ())?;
    let ldb = i32::try_from(b_info.ld).map_err(|_| ())?;
    let ldc = i32::try_from(c_info.ld).map_err(|_| ())?;
    let ldd = i32::try_from(d_info.ld).map_err(|_| ())?;

    let c_ptr = if c.is_null() { d as *const ::core::ffi::c_void } else { c };
    let (c_type_eff, ldc_eff) = if c.is_null() {
        (d_type, ldd)
    } else {
        (c_type, ldc)
    };

    crate::rocblas::gemm(
        ta,
        tb,
        m,
        n,
        k,
        alpha,
        a,
        a_type,
        lda,
        b,
        b_type,
        ldb,
        beta,
        c_ptr,
        c_type_eff,
        ldc_eff,
        d,
        d_type,
        ldd,
        compute_type,
        stream.0.cast(),
    )?;

    if desc.epilogue == hipblasLtEpilogue_t::HIPBLASLT_EPILOGUE_DEFAULT.0 as i32 {
        return Ok(());
    }
    if desc.epilogue == hipblasLtEpilogue_t::HIPBLASLT_EPILOGUE_BIAS.0 as i32 {
        let bias = desc.bias_ptr as *const ::core::ffi::c_void;
        if bias.is_null() {
            return Err(());
        }
        if desc.bias_dtype >= 0 && desc.bias_dtype != d_info.dtype {
            return Err(());
        }
        return crate::rocblas::add_bias(
            m,
            n,
            bias,
            d,
            d_type,
            ldd,
            compute_type,
            stream.0.cast(),
        );
    }

    Err(())
}

pub(crate) fn matmul_algo_get_heuristic(
    light_handle: &Handle,
    operation_desc: hipblasLtMatmulDesc_t,
    a_desc: hipblasLtMatrixLayout_t,
    b_desc: hipblasLtMatrixLayout_t,
    c_desc: hipblasLtMatrixLayout_t,
    d_desc: hipblasLtMatrixLayout_t,
    preference: hipblasLtMatmulPreference_t,
    requested_algo_count: ::core::ffi::c_int,
    heuristic_results_array: &mut cublasLtMatmulHeuristicResult_t,
    return_algo_count: &mut ::core::ffi::c_int,
) -> cublasStatus_t {
    let requested = requested_algo_count.max(1) as usize;
    let force_rocblas = descs()
        .lock()
        .unwrap()
        .get(&(operation_desc.0 as usize))
        .map(|d| d.force_rocblas)
        .unwrap_or(false);
    if (force_rocblas || HIPBLASLT_UNAVAILABLE.load(std::sync::atomic::Ordering::Relaxed))
        && requested_algo_count >= 1
    {
        heuristic_results_array.algo = cublasLtMatmulAlgo_t {
            data: [FALLBACK_ALGO_MAGIC, 0, 0, 0, 0, 0, 0, 0],
        };
        heuristic_results_array.workspaceSize = 0;
        heuristic_results_array.state = cublasStatus_t::SUCCESS;
        heuristic_results_array.wavesCount = 0.0;
        *return_algo_count = 1;
        return Ok(());
    }
    let mut hip_algos = vec![unsafe { mem::zeroed() }; requested];
    let status = unsafe {
        hipblaslt()?.hipblasLtMatmulAlgoGetHeuristic(
            light_handle.handle,
            operation_desc,
            a_desc,
            b_desc,
            c_desc,
            d_desc,
            preference,
            requested_algo_count,
            hip_algos.as_mut_ptr(),
            return_algo_count,
        )
    };
    if !status.is_ok() || *return_algo_count == 0 {
        HIPBLASLT_UNAVAILABLE.store(true, std::sync::atomic::Ordering::Relaxed);
    }
    if status.is_ok() && *return_algo_count > 0 {
        if *return_algo_count as usize > hip_algos.len() {
            return cublasStatus_t::ERROR_INTERNAL_ERROR;
        }
        for (idx, hip_algo) in hip_algos
            .into_iter()
            .take(*return_algo_count as usize)
            .enumerate()
        {
            let heuristic_results_array: *mut cublasLtMatmulHeuristicResult_t = heuristic_results_array;
            let result = unsafe { &mut *heuristic_results_array.add(idx) };
            result.algo = cuda_algo_from_hip(hip_algo.algo);
            result.workspaceSize = hip_algo.workspaceSize;
            result.state = hip_algo.state.map_err(|e| cublasError_t::from(e));
            result.wavesCount = hip_algo.wavesCount;
        }
        return Ok(());
    }
    // hipBLASLt has no usable kernels for this device (e.g. gfx1010/RDNA1).
    // Emit a synthetic algo; cublasLtMatmul will route it to rocBLAS.
    if requested_algo_count >= 1 {
        heuristic_results_array.algo = cublasLtMatmulAlgo_t {
            data: [FALLBACK_ALGO_MAGIC, 0, 0, 0, 0, 0, 0, 0],
        };
        heuristic_results_array.workspaceSize = 0;
        heuristic_results_array.state = cublasStatus_t::SUCCESS;
        heuristic_results_array.wavesCount = 0.0;
        *return_algo_count = 1;
        return Ok(());
    }
    status?;
    Ok(())
}

pub(crate) fn matmul_desc_create(
    matmul_desc: &mut hipblasLtMatmulDesc_t,
    compute_type: hipblasComputeType_t,
    scale_type: hipDataType,
) -> cublasStatus_t {
    let requested_compute = compute_type.0 as i32;
    // hipBLASLt on some ROCm versions rejects fp16 compute descriptors, and it
    // has no gfx1010 kernels anyway. Create a plain fp32 descriptor and force
    // cublasLtMatmul to use the rocBLAS fallback, which honours fp16 compute.
    let force_rocblas = requested_compute == hipblasComputeType_t::HIPBLAS_COMPUTE_16F.0 as i32;
    let create_compute = if force_rocblas {
        hipblasComputeType_t::HIPBLAS_COMPUTE_32F
    } else {
        compute_type
    };
    let create_scale = if force_rocblas {
        hipDataType::HIP_R_32F
    } else {
        scale_type
    };
    unsafe {
        hipblaslt()?.hipblasLtMatmulDescCreate(matmul_desc, create_compute, create_scale)
    }?;
    descs().lock().unwrap().insert(
        matmul_desc.0 as usize,
        DescInfo {
            compute_type: requested_compute,
            scale_type: scale_type.0 as i32,
            trans_a: hipblasOperation_t::HIPBLAS_OP_N.0 as i32,
            trans_b: hipblasOperation_t::HIPBLAS_OP_N.0 as i32,
            epilogue: hipblasLtEpilogue_t::HIPBLASLT_EPILOGUE_DEFAULT.0 as i32,
            bias_ptr: 0,
            bias_dtype: -1,
            force_rocblas,
        },
    );
    Ok(())
}

pub(crate) fn matmul_desc_destroy(matmul_desc: hipblasLtMatmulDesc_t) -> cublasStatus_t {
    unsafe { hipblaslt()?.hipblasLtMatmulDescDestroy(matmul_desc) }?;
    descs().lock().unwrap().remove(&(matmul_desc.0 as usize));
    Ok(())
}

pub(crate) fn matmul_desc_set_attribute(
    matmul_desc: hipblasLtMatmulDesc_t,
    attr: cublasLtMatmulDescAttributes_t,
    buf: *const ::core::ffi::c_void,
    size_in_bytes: usize,
) -> cublasStatus_t {
    if attr == cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_SCALE_TYPE {
        if size_in_bytes != 4 {
            return cublasStatus_t::ERROR_INVALID_VALUE;
        }
        let scale_type = cudaDataType_t(unsafe { *buf.cast() });
        if scale_type != cudaDataType_t::CUDA_R_32F && scale_type != cudaDataType_t::CUDA_R_16F {
            return cublasStatus_t::ERROR_NOT_SUPPORTED;
        }
        if let Some(desc) = descs().lock().unwrap().get_mut(&(matmul_desc.0 as usize)) {
            desc.scale_type = scale_type.0 as i32;
        }
        return Ok(());
    }
    let hip_attr = FromCuda::<_, cublasError_t>::from_cuda(&attr)?;
    match hip_attr {
        hipblasLtMatmulDescAttributes_t::HIPBLASLT_MATMUL_DESC_TRANSA => {
            convert_and_set_attribute::<cublasOperation_t, hipblasOperation_t>(
                matmul_desc,
                buf,
                hip_attr,
            )?;
            if size_in_bytes >= 4 && !buf.is_null() {
                let cuda_op = unsafe { *buf.cast::<u32>() };
                let hip_op = match cuda_op {
                    0 => hipblasOperation_t::HIPBLAS_OP_N.0 as i32,
                    1 => hipblasOperation_t::HIPBLAS_OP_T.0 as i32,
                    2 => hipblasOperation_t::HIPBLAS_OP_C.0 as i32,
                    _ => return cublasStatus_t::ERROR_INVALID_VALUE,
                };
                if let Some(desc) = descs().lock().unwrap().get_mut(&(matmul_desc.0 as usize)) {
                    desc.trans_a = hip_op;
                }
            }
        }
        hipblasLtMatmulDescAttributes_t::HIPBLASLT_MATMUL_DESC_TRANSB => {
            convert_and_set_attribute::<cublasOperation_t, hipblasOperation_t>(
                matmul_desc,
                buf,
                hip_attr,
            )?;
            if size_in_bytes >= 4 && !buf.is_null() {
                let cuda_op = unsafe { *buf.cast::<u32>() };
                let hip_op = match cuda_op {
                    0 => hipblasOperation_t::HIPBLAS_OP_N.0 as i32,
                    1 => hipblasOperation_t::HIPBLAS_OP_T.0 as i32,
                    2 => hipblasOperation_t::HIPBLAS_OP_C.0 as i32,
                    _ => return cublasStatus_t::ERROR_INVALID_VALUE,
                };
                if let Some(desc) = descs().lock().unwrap().get_mut(&(matmul_desc.0 as usize)) {
                    desc.trans_b = hip_op;
                }
            }
        }
        hipblasLtMatmulDescAttributes_t::HIPBLASLT_MATMUL_DESC_EPILOGUE => {
            convert_and_set_attribute::<cublasLtEpilogue_t, hipblasLtEpilogue_t>(
                matmul_desc,
                buf,
                hip_attr,
            )?;
            if size_in_bytes >= 4 && !buf.is_null() {
                let epilogue = unsafe { *buf.cast::<u32>() } as i32;
                if let Some(desc) = descs().lock().unwrap().get_mut(&(matmul_desc.0 as usize)) {
                    desc.epilogue = epilogue;
                }
            }
        }
        hipblasLtMatmulDescAttributes_t::HIPBLASLT_MATMUL_DESC_BIAS_DATA_TYPE => {
            convert_and_set_attribute::<cudaDataType, hipDataType>(matmul_desc, buf, hip_attr)?;
            if size_in_bytes >= 4 && !buf.is_null() {
                let dtype = unsafe { *buf.cast::<u32>() } as i32;
                if let Some(desc) = descs().lock().unwrap().get_mut(&(matmul_desc.0 as usize)) {
                    desc.bias_dtype = dtype;
                }
            }
        }
        hipblasLtMatmulDescAttributes_t::HIPBLASLT_MATMUL_DESC_BIAS_POINTER => {
            unsafe {
                hipblaslt()?.hipblasLtMatmulDescSetAttribute(matmul_desc, hip_attr, buf, size_in_bytes)
            }?;
            if size_in_bytes >= 8 && !buf.is_null() {
                let bias_ptr = unsafe { *buf.cast::<usize>() };
                if let Some(desc) = descs().lock().unwrap().get_mut(&(matmul_desc.0 as usize)) {
                    desc.bias_ptr = bias_ptr;
                }
            }
        }
        _ => return cublasStatus_t::ERROR_NOT_SUPPORTED,
    }
    Ok(())
}

fn convert_and_set_attribute<'a, CudaType: 'a, HipType>(
    matmul_desc: hipblasLtMatmulDesc_t,
    buf: *const std::ffi::c_void,
    hip_attr: hipblasLtMatmulDescAttributes_t,
) -> Result<(), cublasError_t>
where
    HipType: FromCuda<'a, CudaType, cuda_types::cublas::cublasError_t>,
{
    let cublas_operation: &CudaType =
        unsafe { buf.cast::<CudaType>().as_ref() }.ok_or(cublasError_t::INVALID_VALUE)?;
    let hip_operation: HipType = FromCuda::<_, cublasError_t>::from_cuda(cublas_operation)?;
    let hip_buf: *const HipType = &hip_operation;
    unsafe {
        hipblaslt()?.hipblasLtMatmulDescSetAttribute(
            matmul_desc,
            hip_attr,
            hip_buf.cast(),
            mem::size_of::<HipType>(),
        )
    }?;
    Ok(())
}

pub(crate) fn matmul_preference_create(pref: &mut hipblasLtMatmulPreference_t) -> cublasStatus_t {
    unsafe { hipblaslt()?.hipblasLtMatmulPreferenceCreate(pref) }?;
    Ok(())
}

pub(crate) fn matmul_preference_destroy(pref: hipblasLtMatmulPreference_t) -> cublasStatus_t {
    unsafe { hipblaslt()?.hipblasLtMatmulPreferenceDestroy(pref) }?;
    Ok(())
}

pub(crate) fn matmul_preference_set_attribute(
    pref: hipblasLtMatmulPreference_t,
    attr: cublasLtMatmulPreferenceAttributes_t,
    buf: *const ::core::ffi::c_void,
    size_in_bytes: usize,
) -> cublasStatus_t {
    match attr {
        cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES
        | cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES
        | cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES
        | cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES => {
            return Ok(());
        }
        _ => {}
    }
    let attr = FromCuda::<_, cublasError_t>::from_cuda(&attr)?;
    unsafe { hipblaslt()?.hipblasLtMatmulPreferenceSetAttribute(pref, attr, buf, size_in_bytes) }?;
    Ok(())
}

pub(crate) fn matrix_layout_create(
    mat_layout: &mut hipblasLtMatrixLayout_t,
    type_: hipDataType,
    rows: u64,
    cols: u64,
    ld: i64,
) -> cublasStatus_t {
    unsafe { hipblaslt()?.hipblasLtMatrixLayoutCreate(mat_layout, type_, rows, cols, ld) }?;
    layouts().lock().unwrap().insert(
        mat_layout.0 as usize,
        LayoutInfo {
            dtype: type_.0 as i32,
            rows,
            cols,
            ld,
            order: hipblasLtOrder_t::HIPBLASLT_ORDER_COL.0 as i32,
            batch_count: 1,
        },
    );
    Ok(())
}

pub(crate) fn matrix_layout_destroy(mat_layout: hipblasLtMatrixLayout_t) -> cublasStatus_t {
    unsafe { hipblaslt()?.hipblasLtMatrixLayoutDestroy(mat_layout) }?;
    layouts().lock().unwrap().remove(&(mat_layout.0 as usize));
    Ok(())
}

pub(crate) fn matrix_layout_set_attribute(
    mat_layout: hipblasLtMatrixLayout_t,
    attr: hipblasLtMatrixLayoutAttribute_t,
    buf: *const ::core::ffi::c_void,
    size_in_bytes: usize,
) -> cublasStatus_t {
    unsafe {
        hipblaslt()?.hipblasLtMatrixLayoutSetAttribute(mat_layout, attr, buf, size_in_bytes)
    }?;
    if buf.is_null() {
        return Ok(());
    }
    let mut map = layouts().lock().unwrap();
    let Some(info) = map.get_mut(&(mat_layout.0 as usize)) else {
        return Ok(());
    };
    match attr {
        hipblasLtMatrixLayoutAttribute_t::HIPBLASLT_MATRIX_LAYOUT_ROWS if size_in_bytes >= 8 => {
            info.rows = unsafe { *buf.cast::<u64>() };
        }
        hipblasLtMatrixLayoutAttribute_t::HIPBLASLT_MATRIX_LAYOUT_COLS if size_in_bytes >= 8 => {
            info.cols = unsafe { *buf.cast::<u64>() };
        }
        hipblasLtMatrixLayoutAttribute_t::HIPBLASLT_MATRIX_LAYOUT_LD if size_in_bytes >= 8 => {
            info.ld = unsafe { *buf.cast::<i64>() };
        }
        hipblasLtMatrixLayoutAttribute_t::HIPBLASLT_MATRIX_LAYOUT_ORDER if size_in_bytes >= 4 => {
            info.order = unsafe { *buf.cast::<i32>() };
        }
        hipblasLtMatrixLayoutAttribute_t::HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT
            if size_in_bytes >= 4 =>
        {
            info.batch_count = unsafe { *buf.cast::<i32>() };
        }
        hipblasLtMatrixLayoutAttribute_t::HIPBLASLT_MATRIX_LAYOUT_TYPE if size_in_bytes >= 4 => {
            info.dtype = unsafe { *buf.cast::<i32>() };
        }
        _ => {}
    }
    Ok(())
}
