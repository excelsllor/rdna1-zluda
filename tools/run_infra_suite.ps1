# Phase-1 acceptance runner for the ZLUDA/RDNA1 stack.
#
# Runs the infrastructure probe, the training smoke suite and the ecosystem
# shim smoke under the ZLUDA launcher, saves the logs under bench/results/infra
# and prints a one-line summary per suite.
#
# Usage: powershell -File tools\run_infra_suite.ps1 [-SkipProbe] [-SkipTrain] [-SkipShims]
param(
    [switch]$SkipProbe,
    [switch]$SkipTrain,
    [switch]$SkipShims
)

$ErrorActionPreference = "Continue"
$repo = Split-Path -Parent $PSScriptRoot
$outDir = Join-Path $repo "bench\results\infra"
New-Item -ItemType Directory -Force $outDir | Out-Null
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$failures = @()

function Run-Suite([string]$name, [string]$script, [string]$extraArgs) {
    $log = Join-Path $outDir "$name-$stamp.log"
    $cmd = "G:\zluda\run-zluda.cmd venv\Scripts\python.exe `"$repo\bench\$script`" $extraArgs > `"$log`" 2>&1"
    Write-Host "=== $name ==="
    $p = Start-Process -FilePath cmd.exe -ArgumentList '/c', $cmd -PassThru -WindowStyle Hidden
    $done = $p | Wait-Process -Timeout 900 -ErrorAction SilentlyContinue
    if (-not $done -and -not $p.HasExited) {
        Get-Process python,zluda -ErrorAction SilentlyContinue |
            Where-Object { $_.StartTime -gt (Get-Date).AddMinutes(-16) } |
            Stop-Process -Force
        $failures += "$name (timeout)"
        Write-Host "TIMEOUT: $name"
        return
    }
    $summary = Select-String -Path $log -Pattern "SUMMARY" -ErrorAction SilentlyContinue |
        Select-Object -Last 1
    if ($summary) {
        Write-Host $summary.Line.Trim()
        if ($summary.Line -notmatch "(\d+)/\1 ") {
            # not all passed - keep a note; individual failures are in the log
            $fail = Select-String -Path $log -Pattern "^FAIL" -ErrorAction SilentlyContinue
            if ($fail) { $failures += "$name (see $log)" }
        }
    } else {
        $failures += "$name (no SUMMARY line)"
        Write-Host "no SUMMARY line, see $log"
    }
}

if (-not $SkipProbe) { Run-Suite "infra_probe" "infra_probe.py" "--out bench\results\infra\probe.json" }
if (-not $SkipTrain) { Run-Suite "train_suite" "train_suite.py" "" }
if (-not $SkipShims) { Run-Suite "shim_smoke" "shim_smoke.py" "" }

Write-Host ""
if ($failures.Count -eq 0) {
    Write-Host "PHASE-1 INFRA SUITE: PASS ($stamp)"
    exit 0
} else {
    Write-Host "PHASE-1 INFRA SUITE: FAILURES: $($failures -join '; ') (logs in $outDir)"
    exit 1
}
