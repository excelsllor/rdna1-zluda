# Restart the AMD display device that ZLUDA uses.
#
# Run when CUDA init starts hanging (torch.cuda.is_available() blocks, even a
# plain hipStreamCreateWithFlags hangs). This is usually caused by force-killing
# a process during CUDA initialization. Needs an elevated PowerShell.
#
# The default instance id is the Radeon RX 5700 XT used for development; pass
# your own PCI id (Get-PnpDevice -Class Display) when it differs.
param(
    [string]$InstanceId = 'PCI\VEN_1002&DEV_731F&SUBSYS_51011849&REV_C1\6&CF24139&0&00000009',
    [string]$LogPath = (Join-Path $PSScriptRoot 'gpu-reset.log')
)

"start $(Get-Date -Format o) device=$InstanceId" | Out-File -FilePath $LogPath -Encoding utf8
if (Get-Command Restart-PnpDevice -ErrorAction SilentlyContinue) {
    Restart-PnpDevice -InstanceId $InstanceId -Confirm:$false -ErrorAction Continue *>&1 |
        Out-File -FilePath $LogPath -Append -Encoding utf8
} else {
    & pnputil.exe /restart-device $InstanceId *>&1 |
        Out-File -FilePath $LogPath -Append -Encoding utf8
}
"done $(Get-Date -Format o) exit=$LASTEXITCODE" | Out-File -FilePath $LogPath -Append -Encoding utf8
Start-Sleep -Seconds 3
