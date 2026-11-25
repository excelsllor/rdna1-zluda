param(
    [Parameter(Mandatory = $true)][string]$Command,
    [string]$LogPath = "G:\zluda\watchdog\watchdog-$(Get-Date -Format 'yyyyMMdd-HHmmss').log",
    [string]$WatchLog = "",
    [int]$PollSeconds = 30,
    [int]$StallLimit = 2,
    [int]$MaxMinutes = 30,
    [string]$InstanceId = "PCI\VEN_1002&DEV_731F&SUBSYS_51011849&REV_C1\6&CF24139&0&00000009",
    [switch]$AllowDriverRestart
)

# Watchdog for risky GPU workloads on the 8 GB RX 5700 XT under ZLUDA.
#
# Starts $Command through cmd.exe, polls every $PollSeconds and reports:
#   * is the workload process still alive,
#   * display device PnP status,
#   * whether the workload log grew (StallLimit polls without growth = hang).
# On a hang/crash it kills the workload tree. Driver restart is DISABLED by
# default: restarting the active display adapter crashed Windows on 2026-11-25
# (verified), so it is only attempted with -AllowDriverRestart from an elevated
# shell - and that is still risky. Without it the watchdog only kills, which is
# enough in most cases; a manual reboot is the fallback when the GPU is wedged.

$ErrorActionPreference = "Continue"
if (-not $WatchLog) {
    $WatchLog = [System.IO.Path]::ChangeExtension($LogPath, ".watchdog.log")
}
New-Item -ItemType Directory -Force (Split-Path $LogPath) | Out-Null
New-Item -ItemType Directory -Force (Split-Path $WatchLog) | Out-Null

function Write-Watch([string]$msg) {
    $line = "$(Get-Date -Format 'HH:mm:ss') $msg"
    Add-Content -Path $WatchLog -Value $line -Encoding utf8
    Write-Host $line
}

function Stop-Tree([int]$rootPid) {
    $children = Get-CimInstance Win32_Process | Where-Object { $_.ParentProcessId -eq $rootPid }
    foreach ($c in $children) { Stop-Tree $c.ProcessId }
    try { Stop-Process -Id $rootPid -Force -ErrorAction Stop } catch { }
}

function Get-DeviceStatus {
    try {
        $d = Get-PnpDevice -Class Display -ErrorAction SilentlyContinue |
            Where-Object { $_.InstanceId -eq $InstanceId } | Select-Object -First 1
        if ($d) { return [string]$d.Status } else { return "NOT_FOUND" }
    } catch {
        return "QUERY_FAILED"
    }
}

function Restart-Device {
    if (-not $AllowDriverRestart) {
        Write-Watch "driver restart disabled (default): killed only; reboot manually if the GPU stays wedged"
        return $false
    }
    $identity = [System.Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object System.Security.Principal.WindowsPrincipal($identity)
    if (-not $principal.IsInRole([System.Security.Principal.WindowsBuiltInRole]::Administrator)) {
        Write-Watch "driver restart skipped: not elevated (restarting the display adapter can crash Windows)"
        return $false
    }
    Write-Watch "attempting display-device restart ($InstanceId) - this can crash Windows"
    try {
        Restart-PnpDevice -InstanceId $InstanceId -Confirm:$false -ErrorAction Stop
        Start-Sleep -Seconds 5
        Write-Watch "display device restarted, status=$(Get-DeviceStatus)"
        return $true
    } catch {
        Write-Watch "display-device restart failed: $($_.Exception.Message.Trim())"
    }
    return $false
}

Write-Watch "START command=$Command log=$LogPath poll=${PollSeconds}s stall=${StallLimit} max=${MaxMinutes}min restart_allowed=$([int]$AllowDriverRestart)"
$proc = Start-Process -FilePath "cmd.exe" `
    -ArgumentList '/c', "$Command > `"$LogPath`" 2>&1" `
    -PassThru -WindowStyle Hidden
Write-Watch "pid=$($proc.Id)"

$start = Get-Date
$lastSize = -1
$stall = 0
$reason = "completed"
while ($true) {
    Start-Sleep -Seconds $PollSeconds
    $exited = $proc.HasExited
    $logSize = if (Test-Path $LogPath) { (Get-Item $LogPath).Length } else { 0 }
    $dev = Get-DeviceStatus
    $grew = $logSize -gt $lastSize
    if ($grew) { $stall = 0 } else { $stall++ }
    $lastSize = $logSize
    Write-Watch "poll pid=$($proc.Id) alive=$([int](-not $exited)) device=$dev logBytes=$logSize stall=$stall"

    if ($exited) {
        $reason = "completed(exit=$($proc.ExitCode))"
        break
    }
    if ($dev -ne "OK") {
        $reason = "device-status-$dev"
        break
    }
    if ($stall -ge $StallLimit) {
        $reason = "hang(log stalled for $($stall * $PollSeconds)s)"
        break
    }
    if (((Get-Date) - $start).TotalMinutes -ge $MaxMinutes) {
        $reason = "max-time(${MaxMinutes}min)"
        break
    }
}

if (-not $proc.HasExited) {
    Write-Watch "killing workload tree (reason=$reason)"
    Stop-Tree $proc.Id
    Start-Sleep -Seconds 3
    Write-Watch "after kill: alive=$([int](-not $proc.HasExited)) device=$(Get-DeviceStatus)"
    Restart-Device | Out-Null
} else {
    Write-Watch "workload exited: $reason"
}

Write-Watch "END reason=$reason"
exit 0
