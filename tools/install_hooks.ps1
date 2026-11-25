<#
.SYNOPSIS
    Install a local git hook that keeps the exported patch consistent.

.DESCRIPTION
    Adds a check to .git/hooks/pre-push that runs tools/check_patch.ps1, so a
    stale or broken local-patches/zluda-rdna1-gfx1010.patch blocks the push
    instead of reaching GitHub. Existing hook content (for example the git-lfs
    pre-push hook) is preserved: the check is appended, not overwritten.
    Re-run this script after cloning; hooks are not versioned.

.EXAMPLE
    tools/install_hooks.ps1
#>

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$hooksDir = Join-Path $repo '.git\hooks'
if (-not (Test-Path -LiteralPath $hooksDir)) {
    throw "'.git/hooks' not found; run this script inside the repository"
}

$hookPath = Join-Path $hooksDir 'pre-push'
$marker = '# zluda: validate the exported RDNA1 patch'
$block = @"

$marker
root=`$(git rev-parse --show-toplevel) || exit 1
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "`$root/tools/check_patch.ps1" || exit 1
"@

$existing = ''
if (Test-Path -LiteralPath $hookPath) {
    $existing = [System.IO.File]::ReadAllText($hookPath)
    if ($existing.Contains($marker)) {
        "already installed in $hookPath"
        return
    }
}

$content = if ($existing.Length -gt 0) {
    $existing.TrimEnd([char[]]"`r`n") + $block
} else {
    "#!/bin/sh" + $block
}
[System.IO.File]::WriteAllText($hookPath, $content.Replace("`r`n", "`n"), (New-Object System.Text.UTF8Encoding($false)))
"installed patch check in $hookPath"
