# GPU CI runner for the ZLUDA/RDNA1 stack (audit item P1).
#
# Runs the regression gate under the safe ZLUDA launcher, stores logs, and exits
# non-zero on failure so it can be wired into CI / pre-commit hooks.
#
# Safe defaults: crashy SDPA fast backends are disabled by zluda_ops.auto, the
# gate only uses the well-tested suites (phase-2 kernels, shim smoke, training),
# and never touches llama.cpp MMQ/FA or raw PyTorch SDPA selection.
#
# Usage:
#   powershell -NoProfile -ExecutionPolicy Bypass -File tools\run_gpu_ci.ps1 [-Update]
param([switch]$Update)
$ErrorActionPreference = "Continue"
$repo = Split-Path -Parent $PSScriptRoot
$outDir = Join-Path $repo "bench\results\infra"
New-Item -ItemType Directory -Force $outDir | Out-Null
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$log = Join-Path $outDir "gpu_ci-$stamp.log"
$args2 = "bench\regression_gate.py --tag ci-$stamp"
if ($Update) { $args2 += " --update" }
$bat = Join-Path $env:TEMP "gpu_ci_$stamp.cmd"
$lines = @('@echo off',
           "call `"$repo\run-zluda.cmd`" venv\Scripts\python.exe `"$repo\$args2`" > `"$log`" 2>&1",
           'exit /b %ERRORLEVEL%')
Set-Content -Path $bat -Value $lines -Encoding ascii
Write-Host "GPU CI: running $args2"
$p = Start-Process -FilePath $bat -PassThru -WindowStyle Hidden
$p | Wait-Process -Timeout 2400 -ErrorAction SilentlyContinue | Out-Null
if (-not $p.HasExited) {
    Get-Process python,zluda -ErrorAction SilentlyContinue |
        Where-Object { $_.StartTime -gt (Get-Date).AddMinutes(-42) } |
        Stop-Process -Force
    Write-Host "GPU CI: TIMEOUT (log $log)"
    exit 1
}
Get-Content $log -ErrorAction SilentlyContinue | Select-Object -Last 12 |
    ForEach-Object { Write-Host "  $_" }
if ($p.ExitCode -eq 0) {
    Write-Host "GPU CI: PASS (log $log)"
    exit 0
}
Write-Host "GPU CI: FAIL (exit $($p.ExitCode), log $log)"
exit 1
