use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::{Mutex, OnceLock};

pub(crate) const OP_N: i32 = 111;
pub(crate) const OP_T: i32 = 112;

pub(crate) const DT_F16: i32 = 150;
pub(crate) const DT_F32: i32 = 151;
pub(crate) const DT_F64: i32 = 152;
pub(crate) const DT_I8: i32 = 160;
pub(crate) const DT_I32: i32 = 162;
pub(crate) const DT_BF16: i32 = 168;

type RocblasHandle = *mut c_void;

type CreateHandle = unsafe extern "C" fn(*mut RocblasHandle) -> i32;
type DestroyHandle = unsafe extern "C" fn(RocblasHandle) -> i32;
type SetStream = unsafe extern "C" fn(RocblasHandle, *mut c_void) -> i32;
type GemmEx = unsafe extern "C" fn(
    RocblasHandle,
    i32,
    i32,
    i32,
    i32,
    i32,
    *const c_void,
    *const c_void,
    i32,
    i32,
    *const c_void,
    i32,
    i32,
    *const c_void,
    *const c_void,
    i32,
    i32,
    *mut c_void,
    i32,
    i32,
    i32,
    i32,
    i32,
    u32,
) -> i32;

struct Rocblas {
    _lib: libloading::os::windows::Library,
    create_handle: CreateHandle,
    destroy_handle: DestroyHandle,
    set_stream: SetStream,
    gemm_ex: GemmEx,
}

impl Rocblas {
    unsafe fn new() -> Result<Self, ()> {
        let hmodule = zluda_windows::try_load_from_self_or_hip("rocblas.dll").ok_or(())?;
        let lib = libloading::os::windows::Library::from_raw(hmodule.0 as _);
        macro_rules! sym {
            ($name:literal, $ty:ty) => {
                *lib.get::<$ty>(concat!($name, "\0").as_bytes())
                    .map_err(|_| ())?
            };
        }
        Ok(Self {
            create_handle: sym!("rocblas_create_handle", CreateHandle),
            destroy_handle: sym!("rocblas_destroy_handle", DestroyHandle),
            set_stream: sym!("rocblas_set_stream", SetStream),
            gemm_ex: sym!("rocblas_gemm_ex", GemmEx),
            _lib: lib,
        })
    }
}

fn rocblas() -> Result<&'static Rocblas, ()> {
    static LOCK: OnceLock<Result<Rocblas, ()>> = OnceLock::new();
    LOCK.get_or_init(|| unsafe { Rocblas::new() })
        .as_ref()
        .map_err(|_| ())
}

thread_local! {
    static THREAD_HANDLE: std::cell::Cell<usize> = std::cell::Cell::new(0);
}

fn thread_handle(rocm: &Rocblas) -> Option<RocblasHandle> {
    THREAD_HANDLE.with(|cell| {
        let mut value = cell.get();
        if value == 0 {
            let mut handle: RocblasHandle = std::ptr::null_mut();
            if unsafe { (rocm.create_handle)(&mut handle) } != 0 {
                return None;
            }
            value = handle as usize;
            cell.set(value);
        }
        Some(value as RocblasHandle)
    })
}

type HipMalloc = unsafe extern "C" fn(*mut *mut c_void, usize) -> i32;
type HipFree = unsafe extern "C" fn(*mut c_void) -> i32;
type HipMemcpy = unsafe extern "C" fn(*mut c_void, *const c_void, usize, i32) -> i32;

struct Hip {
    _lib: libloading::os::windows::Library,
    malloc: HipMalloc,
    free: HipFree,
    memcpy: HipMemcpy,
}

impl Hip {
    unsafe fn new() -> Result<Self, ()> {
        let hmodule = zluda_windows::try_load_from_self_or_hip("amdhip64_7.dll")
            .or_else(|| zluda_windows::try_load_from_self_or_hip("amdhip64_6.dll"))
            .ok_or(())?;
        let lib = libloading::os::windows::Library::from_raw(hmodule.0 as _);
        macro_rules! sym {
            ($name:literal, $ty:ty) => {
                *lib.get::<$ty>(concat!($name, "\0").as_bytes())
                    .map_err(|_| ())?
            };
        }
        Ok(Self {
            malloc: sym!("hipMalloc", HipMalloc),
            free: sym!("hipFree", HipFree),
            memcpy: sym!("hipMemcpy", HipMemcpy),
            _lib: lib,
        })
    }
}

fn hip() -> Result<&'static Hip, ()> {
    static LOCK: OnceLock<Result<Hip, ()>> = OnceLock::new();
    LOCK.get_or_init(|| unsafe { Hip::new() })
        .as_ref()
        .map_err(|_| ())
}

const HIP_MEMCPY_HOST_TO_DEVICE: i32 = 1;

fn elem_size(dtype: i32) -> Option<usize> {
    match dtype {
        DT_F16 | DT_BF16 => Some(2),
        DT_F32 | DT_I32 => Some(4),
        DT_F64 => Some(8),
        DT_I8 => Some(1),
        _ => None,
    }
}

fn one_pattern(dtype: i32) -> Option<&'static [u8]> {
    match dtype {
        DT_F16 => Some(&[0x00, 0x3C]),
        DT_BF16 => Some(&[0x80, 0x3F]),
        DT_F32 => Some(&[0x00, 0x00, 0x80, 0x3F]),
        DT_F64 => Some(&[0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xF0, 0x3F]),
        _ => None,
    }
}

fn ones(dtype: i32, count: u64) -> Result<*mut c_void, ()> {
    if count == 0 {
        return Err(());
    }
    static CACHE: OnceLock<Mutex<HashMap<(i32, u64), usize>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(ptr) = cache.lock().unwrap().get(&(dtype, count)).copied() {
        return Ok(ptr as *mut c_void);
    }
    let size = elem_size(dtype).ok_or(())?;
    let pattern = one_pattern(dtype).ok_or(())?;
    let total = (count as usize).checked_mul(size).ok_or(())?;
    let hip = hip()?;
    let mut ptr: *mut c_void = std::ptr::null_mut();
    if unsafe { (hip.malloc)(&mut ptr, total) } != 0 {
        return Err(());
    }
    let mut host = vec![0u8; total];
    for chunk in host.chunks_exact_mut(size) {
        chunk.copy_from_slice(pattern);
    }
    if unsafe { (hip.memcpy)(ptr, host.as_ptr().cast(), total, HIP_MEMCPY_HOST_TO_DEVICE) } != 0 {
        unsafe { (hip.free)(ptr) };
        return Err(());
    }
    cache.lock().unwrap().insert((dtype, count), ptr as usize);
    Ok(ptr)
}

#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn gemm(
    trans_a: i32,
    trans_b: i32,
    m: i32,
    n: i32,
    k: i32,
    alpha: *const c_void,
    a: *const c_void,
    a_type: i32,
    lda: i32,
    b: *const c_void,
    b_type: i32,
    ldb: i32,
    beta: *const c_void,
    c: *const c_void,
    c_type: i32,
    ldc: i32,
    d: *mut c_void,
    d_type: i32,
    ldd: i32,
    compute_type: i32,
    stream: *mut c_void,
) -> Result<(), ()> {
    let rocm = rocblas()?;
    let handle = thread_handle(rocm).ok_or(())?;
    if (rocm.set_stream)(handle, stream) != 0 {
        return Err(());
    }
    let status = (rocm.gemm_ex)(
        handle, trans_a, trans_b, m, n, k, alpha, a, a_type, lda, b, b_type, ldb, beta, c, c_type,
        ldc, d, d_type, ldd, compute_type, 0, 0, 0,
    );
    if status != 0 {
        return Err(());
    }
    Ok(())
}

/// d += bias(m, 1) * ones(1, n)
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn add_bias(
    m: i32,
    n: i32,
    bias: *const c_void,
    d: *mut c_void,
    dtype: i32,
    ldc: i32,
    compute_type: i32,
    stream: *mut c_void,
) -> Result<(), ()> {
    if bias.is_null() || d.is_null() {
        return Err(());
    }
    let ones = ones(dtype, n as u64)?;
    let mut one_f32 = 1.0f32;
    let mut one_f64 = 1.0f64;
    let mut one_f16: u16 = 0x3C00;
    let mut one_bf16: u16 = 0x3F80;
    let mut one_i32: i32 = 1;
    let alpha: *const c_void = match compute_type {
        DT_F64 => (&one_f64 as *const f64).cast(),
        DT_F16 => (&one_f16 as *const u16).cast(),
        DT_BF16 => (&one_bf16 as *const u16).cast(),
        DT_I32 => (&one_i32 as *const i32).cast(),
        _ => (&one_f32 as *const f32).cast(),
    };
    gemm(
        OP_N,
        OP_N,
        m,
        n,
        1,
        alpha,
        bias,
        dtype,
        m,
        ones,
        dtype,
        1,
        alpha,
        d,
        dtype,
        ldc,
        d,
        dtype,
        ldc,
        compute_type,
        stream,
    )
}
