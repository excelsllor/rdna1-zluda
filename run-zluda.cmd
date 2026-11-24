@echo off
setlocal
set "VS=C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools"
if exist "%VS%\VC\Auxiliary\Build\vcvars64.bat" call "%VS%\VC\Auxiliary\Build\vcvars64.bat" >nul
set "PATH=%VS%\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin;%VS%\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja;%PATH%"

set "HIP_PATH=G:\zluda\hip"
set "HIP_VISIBLE_DEVICES=0"
set "ZLUDA_CC=8.6"
set "TRITON_PTXAS_PATH=G:\zluda\target\release\ptxas.exe"
set "TRITON_CACHE_DIR=G:\zluda\.triton_cache"
set "TORCHINDUCTOR_CACHE_DIR=G:\zluda\.inductor_cache"
set "PATH=G:\zluda\target\release;G:\zluda\hip\bin;%PATH%"
set "ROCBLAS_USE_HIPBLASLT=0"
if exist "%~dp0tensile_overrides\gfx1010_hh.csv" set "ROCBLAS_TENSILE_GEMM_OVERRIDE_PATH=%~dp0tensile_overrides\gfx1010_hh.csv"

"G:\zluda\target\release\zluda.exe" -- %*
