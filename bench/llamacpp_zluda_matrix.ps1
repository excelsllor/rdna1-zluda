# Run llama.cpp (same engine as Ollama) on the ZLUDA GPU and compare with the
# Ollama Vulkan numbers. Requires the patched ZLUDA (cudaLaunchKernelEx/PDL) and
# the CUDA llama.cpp build in tools/llamacpp-cuda (gitignored).
#
# Usage:
#   powershell -File bench/llamacpp_zluda_matrix.ps1
#   powershell -File bench/llamacpp_zluda_matrix.ps1 -Ctx 2048,8192 -Models q4
param(
    [string]$Ctx = "2048,8192,16384,32768",
    [string]$Models = "q4,f16",
    [int]$Predict = 32,
    [string]$Fa = "1,0",
    [int]$Reps = 3
)

$ErrorActionPreference = "Continue"
$repo = "G:\zluda"
$exe = Join-Path $repo "tools\llamacpp-cuda\build\llama-bench.exe"
$cudart = Join-Path $repo "tools\llamacpp-cuda\cudart"
$gguf = Join-Path $repo "models\qwen05-gguf"
$modelFiles = @{
    q4  = Join-Path $gguf "qwen2.5-0.5b-instruct-q4_k_m.gguf"
    f16 = Join-Path $gguf "qwen2.5-0.5b-instruct-fp16.gguf"
}
$outDir = Join-Path $repo "bench\results"
New-Item -ItemType Directory -Force $outDir | Out-Null

$env:PATH = "$(Split-Path $exe);$cudart;$env:PATH"
$env:GGML_CUDA_DISABLE_GRAPHS = "1"

$rows = @()
foreach ($faVal in ($Fa -split ",")) {
    foreach ($which in ($Models -split ",")) {
        $model = $modelFiles[$which]
        if (-not (Test-Path $model)) { Write-Host "missing model $model"; continue }
        $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
        $jsonOut = Join-Path $outDir "llamacpp_zluda_${which}_fa${faVal}-${stamp}.json"
        $args = @("-m", $model, "-p", $Ctx, "-n", "$Predict", "-ngl", "99",
                  "-fa", $faVal, "-r", "$Reps", "-o", "json")
        Write-Host "=== llama.cpp/ZLUDA model=$which fa=$faVal ctx=$Ctx ==="
        $sw = [System.Diagnostics.Stopwatch]::StartNew()
        & (Join-Path $repo "run-zluda.cmd") $exe @args *> $jsonOut
        $code = $LASTEXITCODE
        Write-Host ("exit=$code wall=$([math]::Round($sw.Elapsed.TotalMinutes,1))min -> $jsonOut")
        $rows += [pscustomobject]@{model = $which; fa = $faVal; exit = $code; file = $jsonOut}
    }
}
$manifest = Join-Path $outDir ("llamacpp_zluda_manifest-" + (Get-Date -Format "yyyyMMdd-HHmmss") + ".json")
$rows | ConvertTo-Json -Depth 3 | Set-Content -Path $manifest -Encoding utf8
Write-Host "manifest: $manifest"
