import ctypes
cu = ctypes.WinDLL("nvcuda.dll")
blas = ctypes.WinDLL("cublas64_11.dll")
cu.cuInit.argtypes=[ctypes.c_uint]; cu.cuInit.restype=ctypes.c_int
cu.cuDeviceGet.argtypes=[ctypes.POINTER(ctypes.c_int),ctypes.c_int]; cu.cuDeviceGet.restype=ctypes.c_int
cu.cuCtxCreate_v2.argtypes=[ctypes.POINTER(ctypes.c_void_p),ctypes.c_uint,ctypes.c_int]; cu.cuCtxCreate_v2.restype=ctypes.c_int
cu.cuMemAlloc_v2.argtypes=[ctypes.POINTER(ctypes.c_void_p),ctypes.c_size_t]; cu.cuMemAlloc_v2.restype=ctypes.c_int
blas.cublasCreate_v2.argtypes=[ctypes.POINTER(ctypes.c_void_p)]; blas.cublasCreate_v2.restype=ctypes.c_int
blas.cublasSetWorkspace_v2.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_size_t]; blas.cublasSetWorkspace_v2.restype=ctypes.c_int
print("cuInit", cu.cuInit(0), flush=True)
d=ctypes.c_int(); print("dev", cu.cuDeviceGet(ctypes.byref(d),0), d.value, flush=True)
ctx=ctypes.c_void_p(); print("ctx", cu.cuCtxCreate_v2(ctypes.byref(ctx),0,d.value), flush=True)
h=ctypes.c_void_p(); print("handle", blas.cublasCreate_v2(ctypes.byref(h)), h.value, flush=True)
for size in (0, 1024, 1<<20, 32<<20):
    ws=ctypes.c_void_p(); print("alloc", size, cu.cuMemAlloc_v2(ctypes.byref(ws), max(size,1)), ws.value, flush=True)
    st=blas.cublasSetWorkspace_v2(h, ws, size)
    print("setworkspace size", size, "status", st, flush=True)
