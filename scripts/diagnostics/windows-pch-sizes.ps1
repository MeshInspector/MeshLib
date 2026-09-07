# Report the size of the precompiled headers produced by the MRPch,
# MRMesh and MRViewer projects so PCH growth is visible in the build log.
# Both build systems write under source\TempOutput: MSBuild emits one
# MRPch.pch per project's IntDir, CMake/Ninja emits cmake_pch.hxx.pch
# (only MRPch builds one; MRMesh/MRViewer REUSE_FROM it).
[CmdletBinding()]
param(
    [string]$BuildRoot = (Join-Path $Env:GITHUB_WORKSPACE 'source\TempOutput')
)

if (-not (Test-Path $BuildRoot)) {
    Write-Host "Build output directory '$BuildRoot' does not exist - skipping PCH size report."
    exit 0
}

$files = Get-ChildItem -Path $BuildRoot -Recurse -Filter *.pch -File -ErrorAction SilentlyContinue
if (-not $files) {
    Write-Host "No .pch files found under '$BuildRoot'."
    exit 0
}

$files |
    Sort-Object FullName |
    Select-Object `
        @{ N = 'Size (MB)'; E = { '{0,8:N2}' -f ($_.Length / 1MB) } },
        @{ N = 'Path'; E = { $_.FullName } } |
    Format-Table -AutoSize | Out-String -Width 4096 | Write-Host
