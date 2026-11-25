<#
.SYNOPSIS
    Regenerate local-patches/zluda-rdna1-gfx1010.patch from the fork history.

.DESCRIPTION
    The patch is the diff of the current HEAD against the upstream base
    (v7-preview.21 by default), excluding the patch file itself so that the
    patch never contains itself.

    Output is UTF-8 without BOM and LF line endings: git apply rejects UTF-16
    patches and BOM/CRLF cause noisy failures for other users. Always use this
    script (or tools/check_patch.ps1) instead of writing the patch by hand.

.EXAMPLE
    tools/export_patch.ps1
    tools/export_patch.ps1 -Base v7-preview.21 -Output local-patches/test.patch
#>
param(
    [string]$Base = 'v7-preview.21',
    [string]$Output = 'local-patches/zluda-rdna1-gfx1010.patch'
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot

git -C $repo rev-parse --verify "$Base^{commit}" *> $null
if ($LASTEXITCODE -ne 0) {
    throw "base revision '$Base' was not found; fetch tags/history first (git fetch --tags)"
}

$outPath = Join-Path $repo $Output
$outDir = Split-Path -Parent $outPath
if (-not (Test-Path -LiteralPath $outDir)) {
    New-Item -ItemType Directory -Path $outDir | Out-Null
}

$exclude = ':(exclude)' + ($Output -replace '\\', '/')
git -C $repo diff --binary --full-index "--output=$outPath" "$Base..HEAD" -- . $exclude
if ($LASTEXITCODE -ne 0) {
    throw 'git diff failed'
}

$bytes = [System.IO.File]::ReadAllBytes($outPath)
if ($bytes.Length -eq 0) {
    throw 'the generated patch is empty'
}
"exported $Output ($($bytes.Length) bytes) from $Base..HEAD"
