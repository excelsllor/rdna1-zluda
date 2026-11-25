@echo off
rem Enter the Visual Studio x64 build environment and run the given command.
rem Set VS to a Visual Studio installation path to skip auto-detection.
setlocal enabledelayedexpansion

if not defined VS (
    set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
    if exist "!VSWHERE!" for /f "usebackq tokens=*" %%i in (`"!VSWHERE!" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VS=%%i"
)
if not defined VS if exist "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools" set "VS=C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools"
if not defined VS (
    echo [build-env] Visual Studio with the C++ toolchain was not found.
    echo [build-env] Install VS 2022/2026 Build Tools or set VS to the installation path.
    exit /b 1
)

call "!VS!\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 (
    echo [build-env] vcvars64 failed for "!VS!"
    exit /b 1
)

rem CMake and Ninja bundled with Visual Studio are required by llvm_zluda.
if exist "!VS!\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin" set "PATH=!VS!\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin;!PATH!"
if exist "!VS!\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja" set "PATH=!VS!\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja;!PATH!"

set CARGO_TERM_COLOR=never
%*
