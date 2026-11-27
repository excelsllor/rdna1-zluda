# Cross-backend re-measurement runner: Vulkan vs ZLUDA at max fitting context.
#
# What it does:
#   1. prepares Q8_0 GGUF and the Ollama models (qwen05-q8, qwen05-q4 if missing);
#   2. runs the Vulkan matrix (Ollama/llama.cpp, FA=0, KV f16) over the contexts
#      that fit (up to 32768) for F32/BF16/F16/Q8_0/Q4_K_M;
#   3. runs the ZLUDA PyTorch matrix (no FA, no compile): dtype fp32/fp16/
#      fp16+accum/bf16/bf16-emul and q8/q4 dequantized to fp32, for math and
#      sdpa attention backends, probing contexts up to 32768 and recording the
#      largest that fits;
#   4. summarises everything into docs/ENGINE_COMPARISON_REMATRIX.md.
#
# Usage (after a reboot, the GPU driver must be healthy):
#   powershell -NoProfile -ExecutionPolicy Bypass -File tools\run_cross_backend_rematrix.ps1
#   ... -SkipVulkan   # ZLUDA only
#   ... -SkipZluda    # Vulkan only
param(
    [switch]$SkipVulkan,
    [switch]$SkipZluda,
    [switch]$SkipModelPrep
)
$ErrorActionPreference = "Continue"
$repo = Split-Path -Parent $PSScriptRoot
$py = Join-Path $repo "venv\Scripts\python.exe"
$ggufDir = Join-Path $repo "models\qwen05-gguf"
$hfDir = Join-Path $repo "models\Qwen2.5-0.5B-Instruct"
$vulkanCtx = "2048,8192,32768"
$zludaCtx = "2048,4096,8192,12288,16384,24576,32768"

function Invoke-Run([string]$name, [string]$innerCmd, [int]$timeoutSec = 2700) {
    Write-Host "=== $name ==="
    $bat = Join-Path $env:TEMP "rematrix_$name.cmd"
    $log = Join-Path $env:TEMP "rematrix_$name.log"
    $lines = @('@echo off', "($innerCmd) > `"$log`" 2>&1", 'exit /b %ERRORLEVEL%')
    Set-Content -Path $bat -Value $lines -Encoding ascii
    $p = Start-Process -FilePath $bat -PassThru -WindowStyle Hidden
    $p | Wait-Process -Timeout $timeoutSec -ErrorAction SilentlyContinue | Out-Null
    if (-not $p.HasExited) {
        Get-Process python,zluda -ErrorAction SilentlyContinue |
            Where-Object { $_.StartTime -gt (Get-Date).AddMinutes(-$([int]($timeoutSec / 60) + 2)) } |
            Stop-Process -Force
        Write-Host "TIMEOUT: $name (log $log)"
        return $false
    }
    Get-Content $log -ErrorAction SilentlyContinue | Select-Object -Last 6 |
        ForEach-Object { Write-Host "  $_" }
    if ($p.ExitCode -ne 0) { Write-Host "exit=$($p.ExitCode) (log $log)" }
    return ($p.ExitCode -eq 0)
}

function Run-Python([string]$name, [string]$scriptArgs, [int]$timeoutSec = 2400) {
    $parts = $scriptArgs.Split(' ', 2)
    $inner = "`"$py`" `"$repo\bench\$($parts[0])`" " + $parts[1]
    return Invoke-Run $name $inner $timeoutSec
}

function Run-Zluda([string]$name, [string]$scriptArgs, [int]$timeoutSec = 2400) {
    $parts = $scriptArgs.Split(' ', 2)
    $inner = "call `"$repo\run-zluda.cmd`" venv\Scripts\python.exe `"$repo\bench\$($parts[0])`" " + $parts[1]
    return Invoke-Run $name $inner $timeoutSec
}
# ---------- 1. model preparation ----------
if (-not $SkipModelPrep) {
    $q8Gguf = Join-Path $ggufDir "qwen2.5-0.5b-instruct-q8_0.gguf"
    if (-not (Test-Path $q8Gguf)) {
        Write-Host "=== converting Q8_0 GGUF ==="
        & $py "C:\build\llama.cpp\convert_hf_to_gguf.py" $hfDir --outtype q8_0 --outfile $q8Gguf
        if ($LASTEXITCODE -ne 0) { Write-Host "Q8_0 conversion FAILED (continuing without q8)" }
    }
    $models = (& ollama list 2>$null | Out-String)
    foreach ($spec in @(
        @{name = "qwen05-q8"; gguf = $q8Gguf },
        @{name = "qwen05-q4"; gguf = (Join-Path $ggufDir "qwen2.5-0.5b-instruct-q4_k_m.gguf") }
    )) {
        if (($models -notmatch [regex]::Escape($spec.name)) -and (Test-Path $spec.gguf)) {
            $mf = Join-Path $env:TEMP "Modelfile_$($spec.name)"
            "FROM `"$($spec.gguf)`"`r`nPARAMETER num_ctx 32768" |
                Set-Content -Path $mf -Encoding ascii
            Write-Host "=== importing Ollama model $($spec.name) ==="
            & ollama create $spec.name -f $mf
        }
    }
}

# ---------- 2. Vulkan matrix ----------
if (-not $SkipVulkan) {
    Run-Python "vulkan_matrix" "run_vulkan_matrix.py --models qwen05-f32,qwen05-bf16,qwen05-f16,qwen05-q8,qwen05-q4 --fa 0 --kv f16 --ctx $vulkanCtx --num-predict 32"
}

# ---------- 3. ZLUDA matrix ----------
if (-not $SkipZluda) {
    $configs = @(
        @{tag = 'fp16_math';        args = '--dtype fp16 --attn math' },
        @{tag = 'fp16_sdpa';        args = '--dtype fp16 --attn sdpa_fast' },
        @{tag = 'fp16acc_math';     args = '--dtype fp16 --acc 1 --attn math' },
        @{tag = 'fp16acc_sdpa';     args = '--dtype fp16 --acc 1 --attn sdpa_fast' },
        @{tag = 'fp32_math';        args = '--dtype fp32 --attn math' },
        @{tag = 'fp32_sdpa';        args = '--dtype fp32 --attn sdpa_fast' },
        @{tag = 'bf16_math';        args = '--dtype bf16 --attn math' },
        @{tag = 'bf16_sdpa';        args = '--dtype bf16 --attn sdpa_fast' },
        @{tag = 'bf16emul_math';    args = '--dtype bf16 --bf16-emul --attn math' },
        @{tag = 'bf16emul_sdpa';    args = '--dtype bf16 --bf16-emul --attn sdpa_fast' },
        @{tag = 'q8_fp32_math';     args = '--dtype fp32 --quant int8 --attn math' },
        @{tag = 'q8_fp32_sdpa';     args = '--dtype fp32 --quant int8 --attn sdpa_fast' },
        @{tag = 'q4_fp32_math';     args = '--dtype fp32 --quant int4 --group 64 --attn math' },
        @{tag = 'q4_fp32_sdpa';     args = '--dtype fp32 --quant int4 --group 64 --attn sdpa_fast' }
    )
    foreach ($cfg in $configs) {
        Run-Zluda $cfg.tag ("maxctx_matrix.py {0} --contexts $zludaCtx --decode 8 --tag {1}" -f $cfg.args, $cfg.tag)
    }
}

# ---------- 4. summary ----------
Write-Host "=== summary ==="
& $py (Join-Path $repo "bench\summarize_rematrix.py")
Write-Host "Done: docs/ENGINE_COMPARISON_REMATRIX.md"
