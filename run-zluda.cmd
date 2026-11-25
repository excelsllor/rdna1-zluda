@echo off
rem Run a command through ZLUDA.
rem All paths are relative to this repository; every variable can be overridden
rem from the environment before calling this script.
setlocal

set "ROOT=%~dp0"
set "ROOT=%ROOT:~0,-1%"

rem Visual Studio is only needed for its MSVC runtime/tools at run time. Locate
rem the latest C++ Build Tools/VS installation; keep working when it is absent.
set "VS="
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if exist "%VSWHERE%" for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VS=%%i"
if not defined VS if exist "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools" set "VS=C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools"
if defined VS if exist "%VS%\VC\Auxiliary\Build\vcvars64.bat" call "%VS%\VC\Auxiliary\Build\vcvars64.bat" >nul

if defined ZLUDA_HIP_PATH (set "HIP_PATH=%ZLUDA_HIP_PATH%") else (set "HIP_PATH=%ROOT%\hip")
if not exist "%HIP_PATH%\bin\amdhip64_7.dll" echo [run-zluda] warning: amdhip64_7.dll not found under "%HIP_PATH%\bin"
if not defined HIP_VISIBLE_DEVICES set "HIP_VISIBLE_DEVICES=0"
if not defined ZLUDA_CC set "ZLUDA_CC=8.6"
if not defined TRITON_PTXAS_PATH set "TRITON_PTXAS_PATH=%ROOT%\target\release\ptxas.exe"
if not defined TRITON_CACHE_DIR set "TRITON_CACHE_DIR=%ROOT%\.triton_cache"
if not defined TORCHINDUCTOR_CACHE_DIR set "TORCHINDUCTOR_CACHE_DIR=%ROOT%\.inductor_cache"
set "PATH=%ROOT%\target\release;%HIP_PATH%\bin;%PATH%"
set "ROCBLAS_USE_HIPBLASLT=0"
if not defined ROCBLAS_TENSILE_GEMM_OVERRIDE_PATH if exist "%ROOT%\tensile_overrides\gfx1010_hh.csv" set "ROCBLAS_TENSILE_GEMM_OVERRIDE_PATH=%ROOT%\tensile_overrides\gfx1010_hh.csv"

"%ROOT%\target\release\zluda.exe" -- %*
