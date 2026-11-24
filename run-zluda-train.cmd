@echo off
setlocal
set "ZLUDA_AUTO_OPS=0"
set "ZLUDA_MHA_SLOW=1"
set "ZLUDA_AUTO_VERBOSE=1"
if "%~1"=="" (
  echo usage: run-zluda-train.cmd path\to\script.py [args...]
  exit /b 1
)
call "%~dp0run-zluda.cmd" "%~dp0venv\Scripts\python.exe" "%~dp0tools\run_with_auto.py" %*
