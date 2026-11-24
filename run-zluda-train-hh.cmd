@echo off
setlocal
set "ZLUDA_FP16_ACC=1"
call "%~dp0run-zluda-train.cmd" %*
