@echo off
rem Start ComfyUI on ZLUDA (base mode) and log to comfy-start.log.
call "%~dp0comfy-zluda.cmd" > "%~dp0comfy-start.log" 2>&1
