import ctypes
import sys

path = sys.argv[1]
ptx = open(path, "rb").read() + b"\0"

lib = ctypes.WinDLL("nvcuda.dll")
lib.cuInit.argtypes = [ctypes.c_uint]
lib.cuInit.restype = ctypes.c_int
lib.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
lib.cuDeviceGet.restype = ctypes.c_int
lib.cuCtxCreate_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint, ctypes.c_int]
lib.cuCtxCreate_v2.restype = ctypes.c_int
lib.cuModuleLoadData.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p]
lib.cuModuleLoadData.restype = ctypes.c_int

assert lib.cuInit(0) == 0
dev = ctypes.c_int()
assert lib.cuDeviceGet(ctypes.byref(dev), 0) == 0
ctx = ctypes.c_void_p()
assert lib.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev) == 0
module = ctypes.c_void_p()
err = lib.cuModuleLoadData(ctypes.byref(module), ptx)
print("cuModuleLoadData ->", err)
