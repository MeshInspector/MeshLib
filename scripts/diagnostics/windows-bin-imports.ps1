# Before launching MeshViewer.exe, dump the state the Windows loader will
# actually see. Prints the DLLs present in the current directory, the
# import tables of the key binaries (so we know which DLL names the loader
# wants), and the PATH at launch. Lets DLL-not-found failures self-diagnose
# from the CI log.
#
# Run from the directory containing MeshViewer.exe / MRMesh.dll / MRTest.exe.
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$VcPath
)

Write-Host "=== DLLs / EXEs in $(Get-Location) ==="
Get-ChildItem . -File -Include *.dll,*.exe -Recurse -ErrorAction SilentlyContinue |
    Select-Object Name, @{N='SizeKB';E={[int]($_.Length/1KB)}} |
    Sort-Object Name | Format-Table -AutoSize

# Locate dumpbin.exe via the VC path hint, then fall back to whatever is on
# PATH (Visual Studio Integration step earlier populates MSBuild PATH, not
# VC bin).
$dumpbin = (Get-ChildItem "$VcPath\VC\Tools\MSVC\*\bin\Hostx64\x64\dumpbin.exe" -ErrorAction SilentlyContinue |
            Select-Object -First 1).FullName
if (-not $dumpbin) {
    $dumpbin = (Get-Command dumpbin.exe -ErrorAction SilentlyContinue).Source
}

if ($dumpbin) {
    Write-Host ""
    Write-Host "Using dumpbin at: $dumpbin"
    foreach ($target in 'MeshViewer.exe','MRMesh.dll','MRTest.exe') {
        if (Test-Path $target) {
            Write-Host ""
            Write-Host "=== Import table of $target (dumpbin /dependents) ==="
            & $dumpbin /dependents $target |
                Where-Object { $_ -match '\.dll$' } |
                ForEach-Object { "  " + $_.Trim() }
        }
    }
} else {
    Write-Warning "dumpbin.exe not found; skipping import-table dump"
}

Write-Host ""
Write-Host "=== PATH at diagnostic step ==="
$env:PATH -split ';' | ForEach-Object { "  $_" }
