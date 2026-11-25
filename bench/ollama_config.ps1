# Restart the Ollama server with the requested Vulkan backend configuration.
# Usage: powershell -File bench\ollama_config.ps1 -FlashAttention 1 -KvCacheType f16
param(
    [int]$FlashAttention = 1,
    [string]$KvCacheType = "f16"
)
$ollama = "C:\Users\artyom\AppData\Local\Programs\Ollama\ollama.exe"
Get-Process ollama,"ollama app" -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 2
$env:OLLAMA_FLASH_ATTENTION = "$FlashAttention"
$env:OLLAMA_KV_CACHE_TYPE = $KvCacheType
Start-Process -FilePath $ollama -ArgumentList "serve" -WindowStyle Hidden
for ($i = 0; $i -lt 30; $i++) {
    Start-Sleep -Milliseconds 500
    try {
        $r = Invoke-WebRequest -Uri "http://127.0.0.1:11434/api/tags" -UseBasicParsing -TimeoutSec 2 -ErrorAction Stop
        if ($r.StatusCode -eq 200) {
            "ollama ready (FA=$FlashAttention KV=$KvCacheType)"
            exit 0
        }
    } catch { }
}
"ollama did not become ready"
exit 1