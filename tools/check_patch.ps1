<#
.SYNOPSIS
    Validate local-patches/zluda-rdna1-gfx1010.patch.

.DESCRIPTION
    Checks that the exported patch:
      1. is UTF-8 without BOM and uses LF line endings (required by git apply),
      2. is up to date with the diff of HEAD against the upstream base,
      3. applies cleanly to a pristine tree of the base revision.

    Run this before committing changes to the patch or to the source tree that
    the patch covers. Exit code is non-zero on any failure.

.EXAMPLE
    tools/check_patch.ps1
#>
param(
    [string]$Base = 'v7-preview.21',
    [string]$Patch = 'local-patches/zluda-rdna1-gfx1010.patch'
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$patchPath = Join-Path $repo $Patch

if (-not (Test-Path -LiteralPath $patchPath)) {
    throw "patch '$Patch' is missing"
}

git -C $repo rev-parse --verify "$Base^{commit}" *> $null
if ($LASTEXITCODE -ne 0) {
    throw "base revision '$Base' was not found; fetch tags/history first (git fetch --tags)"
}

# 1. Encoding and line endings.
$bytes = [System.IO.File]::ReadAllBytes($patchPath)
if ($bytes.Length -ge 2 -and $bytes[0] -eq 0xFF -and $bytes[1] -eq 0xFE) {
    throw 'patch is UTF-16 LE; run tools/export_patch.ps1'
}
if ($bytes.Length -ge 3 -and $bytes[0] -eq 0xEF -and $bytes[1] -eq 0xBB -and $bytes[2] -eq 0xBF) {
    throw 'patch has a UTF-8 BOM; run tools/export_patch.ps1'
}
$text = [System.Text.Encoding]::UTF8.GetString($bytes)
if ($text.Contains("`r")) {
    throw 'patch contains CRLF line endings; run tools/export_patch.ps1'
}

$tmp = Join-Path ([System.IO.Path]::GetTempPath()) ('zluda-patch-check-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $tmp | Out-Null
try {
    # 2. Up to date with HEAD.
    $generated = Join-Path $tmp 'generated.patch'
    $exclude = ':(exclude)' + ($Patch -replace '\\', '/')
    git -C $repo diff --binary --full-index "--output=$generated" "$Base..HEAD" -- . $exclude
    if ($LASTEXITCODE -ne 0) {
        throw 'git diff failed'
    }
    $patchHash = (Get-FileHash -LiteralPath $patchPath -Algorithm SHA256).Hash
    $generatedHash = (Get-FileHash -LiteralPath $generated -Algorithm SHA256).Hash
    if ($patchHash -ne $generatedHash) {
        throw "patch is stale against $Base..HEAD; run tools/export_patch.ps1"
    }

    # 3. Applies to a pristine base tree.
    # Use a zip archive + Expand-Archive instead of tar: inside Git Bash (where
    # git hooks run) "tar" resolves to GNU tar, which misparses Windows drive
    # letters as remote hosts.
    $baseDir = Join-Path $tmp 'base'
    New-Item -ItemType Directory -Path $baseDir | Out-Null
    $baseZip = Join-Path $tmp 'base.zip'
    git -C $repo archive --format=zip "--output=$baseZip" $Base
    if ($LASTEXITCODE -ne 0) {
        throw 'git archive failed'
    }
    Expand-Archive -LiteralPath $baseZip -DestinationPath $baseDir -Force
    git -C $baseDir apply --check $generated
    if ($LASTEXITCODE -ne 0) {
        throw "patch does not apply cleanly to $Base"
    }

    "patch OK: UTF-8/LF, matches $Base..HEAD, applies cleanly to $Base"
} finally {
    Remove-Item -LiteralPath $tmp -Recurse -Force -ErrorAction SilentlyContinue
}
