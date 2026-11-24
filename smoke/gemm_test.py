import ctypes
import sys

lib = ctypes.WinDLL("nvcuda.dll")
blas = ctypes.WinDLL("cublas64_12.dll")

lib.cuInit.argtypes = [ctypes.c_uint]
lib.cuInit.restype = ctypes.c_int
lib.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
lib.cuDeviceGet.restype = ctypes.c_int
lib.cuCtxCreate_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint, ctypes.c_int]
lib.cuCtxCreate_v2.restype = ctypes.c_int
lib.cuMemAlloc_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
lib.cuMemAlloc_v2.restype = ctypes.c_int
lib.cuMemcpyHtoD_v2.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
lib.cuMemcpyHtoD_v2.restype = ctypes.c_int
lib.cuMemcpyDtoH_v2.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
lib.cuMemcpyDtoH_v2.restype = ctypes.c_int
lib.cuCtxSynchronize.argtypes = []
lib.cuCtxSynchronize.restype = ctypes.c_int

blas.cublasCreate_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
blas.cublasCreate_v2.restype = ctypes.c_int
blas.cublasSgemm_v2.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_void_p, ctypes.c_int,
    ctypes.c_void_p, ctypes.c_int,
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_void_p, ctypes.c_int,
]
blas.cublasSgemm_v2.restype = ctypes.c_int


def check(err, what):
    if err != 0:
        raise RuntimeError(f"{what} failed with error {err}")


check(lib.cuInit(0), "cuInit")
dev = ctypes.c_int()
check(lib.cuDeviceGet(ctypes.byref(dev), 0), "cuDeviceGet")
ctx = ctypes.c_void_p()
check(lib.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev), "cuCtxCreate")

# Column-major 2x2 matrices: A = [[1, 3], [2, 4]], B = [[5, 7], [6, 8]]
a_host = (ctypes.c_float * 4)(1.0, 2.0, 3.0, 4.0)
b_host = (ctypes.c_float * 4)(5.0, 6.0, 7.0, 8.0)
expected = [23.0, 34.0, 31.0, 46.0]

a_dev = ctypes.c_void_p()
b_dev = ctypes.c_void_p()
c_dev = ctypes.c_void_p()
for buf in (a_dev, b_dev, c_dev):
    check(lib.cuMemAlloc_v2(ctypes.byref(buf), 16), "cuMemAlloc")
check(lib.cuMemcpyHtoD_v2(a_dev, a_host, 16), "cuMemcpyHtoD A")
check(lib.cuMemcpyHtoD_v2(b_dev, b_host, 16), "cuMemcpyHtoD B")

handle = ctypes.c_void_p()
check(blas.cublasCreate_v2(ctypes.byref(handle)), "cublasCreate")

alpha = ctypes.c_float(1.0)
beta = ctypes.c_float(0.0)
status = blas.cublasSgemm_v2(
    handle,
    0, 0,
    2, 2, 2,
    ctypes.byref(alpha),
    a_dev, 2,
    b_dev, 2,
    ctypes.byref(beta),
    c_dev, 2,
)
check(status, "cublasSgemm")
check(lib.cuCtxSynchronize(), "cuCtxSynchronize")

c_host = (ctypes.c_float * 4)()
check(lib.cuMemcpyDtoH_v2(c_host, c_dev, 16), "cuMemcpyDtoH C")
result = [round(v, 4) for v in c_host]
print(f"sgemm result: {result}")
if result != expected:
    print(f"FAIL: expected {expected}")
    sys.exit(1)
print("PASS: cuBLAS (rocBLAS) SGEMM executed on the GPU")
