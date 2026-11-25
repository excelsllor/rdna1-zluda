import ctypes
cu=ctypes.WinDLL("nvcuda.dll"); blas=ctypes.WinDLL("cublas64_11.dll")
cu.cuInit.argtypes=[ctypes.c_uint]; cu.cuInit.restype=ctypes.c_int
cu.cuDeviceGet.argtypes=[ctypes.POINTER(ctypes.c_int),ctypes.c_int]; cu.cuDeviceGet.restype=ctypes.c_int
cu.cuCtxCreate_v2.argtypes=[ctypes.POINTER(ctypes.c_void_p),ctypes.c_uint,ctypes.c_int]; cu.cuCtxCreate_v2.restype=ctypes.c_int
cu.cuMemAlloc_v2.argtypes=[ctypes.POINTER(ctypes.c_void_p),ctypes.c_size_t]; cu.cuMemAlloc_v2.restype=ctypes.c_int
blas.cublasCreate_v2.argtypes=[ctypes.POINTER(ctypes.c_void_p)]; blas.cublasCreate_v2.restype=ctypes.c_int
blas.cublasSetWorkspace_v2.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_size_t]; blas.cublasSetWorkspace_v2.restype=ctypes.c_int
cu.cuInit(0); d=ctypes.c_int(); cu.cuDeviceGet(ctypes.byref(d),0); ctx=ctypes.c_void_p(); cu.cuCtxCreate_v2(ctypes.byref(ctx),0,0)
h=ctypes.c_void_p(); blas.cublasCreate_v2(ctypes.byref(h))
for size in (64<<20,128<<20,256<<20,512<<20,1024<<20):
    ws=ctypes.c_void_p(); st_alloc=cu.cuMemAlloc_v2(ctypes.byref(ws),size); st=blas.cublasSetWorkspace_v2(h,ws,size)
    print(size>>20, "MB alloc",st_alloc,"ws",st, flush=True)
for size in (1024, 1<<20):
    st=blas.cublasSetWorkspace_v2(h,None,size); print("null",size,st,flush=True)
