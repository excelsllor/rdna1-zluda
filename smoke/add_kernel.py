import ctypes
import sys

PTX = b"""
.version 8.0
.target sm_80
.address_size 64

.visible .entry add(.param .u32 a, .param .u32 b, .param .u64 c)
{
    .reg .b32 %r<4>;
    .reg .b64 %rd<2>;
    ld.param.u32 %r1, [a];
    ld.param.u32 %r2, [b];
    add.s32 %r3, %r2, %r1;
    ld.param.u64 %rd1, [c];
    st.global.u32 [%rd1], %r3;
    ret;
}
"""

lib = ctypes.WinDLL("nvcuda.dll")

lib.cuInit.argtypes = [ctypes.c_uint]
lib.cuInit.restype = ctypes.c_int
lib.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
lib.cuDeviceGet.restype = ctypes.c_int
lib.cuCtxCreate_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint, ctypes.c_int]
lib.cuCtxCreate_v2.restype = ctypes.c_int
lib.cuMemAlloc_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
lib.cuMemAlloc_v2.restype = ctypes.c_int
lib.cuModuleLoadData.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p]
lib.cuModuleLoadData.restype = ctypes.c_int
lib.cuModuleGetFunction.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_char_p]
lib.cuModuleGetFunction.restype = ctypes.c_int
lib.cuLaunchKernel.argtypes = [
    ctypes.c_void_p,
    ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
    ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
    ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
]
lib.cuLaunchKernel.restype = ctypes.c_int
lib.cuCtxSynchronize.argtypes = []
lib.cuCtxSynchronize.restype = ctypes.c_int
lib.cuMemcpyHtoD_v2.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
lib.cuMemcpyHtoD_v2.restype = ctypes.c_int
lib.cuMemcpyDtoH_v2.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
lib.cuMemcpyDtoH_v2.restype = ctypes.c_int


def check(err, what):
    if err != 0:
        raise RuntimeError(f"{what} failed with cuda error {err}")


check(lib.cuInit(0), "cuInit")

dev = ctypes.c_int()
check(lib.cuDeviceGet(ctypes.byref(dev), 0), "cuDeviceGet")

ctx = ctypes.c_void_p()
check(lib.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev), "cuCtxCreate")

dptr = ctypes.c_void_p()
check(lib.cuMemAlloc_v2(ctypes.byref(dptr), 4), "cuMemAlloc")

module = ctypes.c_void_p()
check(lib.cuModuleLoadData(ctypes.byref(module), PTX), "cuModuleLoadData")

func = ctypes.c_void_p()
check(lib.cuModuleGetFunction(ctypes.byref(func), module, b"add"), "cuModuleGetFunction")

a = ctypes.c_int(21)
b = ctypes.c_int(21)
args = (ctypes.c_void_p * 3)()
args[0] = ctypes.cast(ctypes.byref(a), ctypes.c_void_p)
args[1] = ctypes.cast(ctypes.byref(b), ctypes.c_void_p)
args[2] = ctypes.cast(ctypes.byref(dptr), ctypes.c_void_p)

check(lib.cuLaunchKernel(func, 1, 1, 1, 1, 1, 1, 0, None, args, None), "cuLaunchKernel")
check(lib.cuCtxSynchronize(), "cuCtxSynchronize")

out = ctypes.c_int(0)
check(lib.cuMemcpyDtoH_v2(ctypes.byref(out), dptr, 4), "cuMemcpyDtoH")

print(f"kernel result: {out.value}")
if out.value != 42:
    print("FAIL: unexpected result")
    sys.exit(1)
print("PASS: ZLUDA executed a CUDA kernel on the GPU")
