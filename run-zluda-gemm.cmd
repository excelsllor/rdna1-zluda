@echo off
rem GEMM-optimized stack: HIP SDK 6.4 + tuned gfx1010 rocBLAS from likelovewant.
rem NOTE: this stack has no MIOpen, so cuDNN/cudnn is unavailable here.
setlocal
set "VS=C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools"
if exist "%VS%\VC\Auxiliary\Build\vcvars64.bat" call "%VS%\VC\Auxiliary\Build\vcvars64.bat" >nul
set "PATH=%VS%\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin;%VS%\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja;%PATH%"

set "HIP_PATH=G:\rocm64"
set "HIP_VISIBLE_DEVICES=0"
set "ZLUDA_CC=8.6"
set "TRITON_PTXAS_PATH=G:\zluda\target\release\ptxas.exe"
set "TRITON_CACHE_DIR=G:\zluda\.triton_cache"
set "TORCHINDUCTOR_CACHE_DIR=G:\zluda\.inductor_cache"
set "PATH=G:\zluda\target\release;G:\rocm64\bin;%PATH%"

"G:\zluda\target\release\zluda.exe" -- %*
