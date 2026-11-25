@echo off
rem Build the HIP GEMV prototype (gfx1010). Requires the portable VS 2022 env
rem and the ROCm clang. Usage: build-env.cmd tools\hip_proto\build_gemv.cmd
set "CLANG=G:\rocm64\bin\clang++.exe"
set "SRC=%~dp0gemv_fp16.cpp"
set "OUT=%~dp0gemv_fp16.dll"
"%CLANG%" -x hip --offload-arch=gfx1010 -O3 -shared -o "%OUT%" "%SRC%" -I"G:\zluda\hip\include" -L"G:\zluda\hip\lib" -lamdhip64
echo BUILD_EXIT=%ERRORLEVEL%