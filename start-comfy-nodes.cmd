@echo off
rem Start ComfyUI on ZLUDA with custom nodes and log to comfy-start-nodes.log.
call "%~dp0comfy-zluda-nodes.cmd" > "%~dp0comfy-start-nodes.log" 2>&1
