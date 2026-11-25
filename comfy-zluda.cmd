@echo off
rem ComfyUI on ZLUDA in base mode (no custom nodes, most reliable).
rem COMFY_DIR   - ComfyUI checkout (default: this machine's layout).
rem COMFY_PYTHON - Python of the ComfyUI venv (default: <repo>\venv-comfy).
setlocal
if not defined COMFY_DIR set "COMFY_DIR=D:\comfyui\Data\Packages\ComfyUI"
if not defined COMFY_PYTHON set "COMFY_PYTHON=%~dp0venv-comfy\Scripts\python.exe"
set "ZLUDA_AUTO_OPS=1"
set "ZLUDA_AUTO_VERBOSE=1"
cd /d "%COMFY_DIR%" || exit /b 1
call "%~dp0run-zluda.cmd" "%COMFY_PYTHON%" main.py --listen 127.0.0.1 --port 8188 --use-pytorch-cross-attention --force-fp16 --preview-method auto --disable-smart-memory --gpu-only --disable-pinned-memory --disable-all-custom-nodes
