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

    # Index of binaries present locally (case-insensitive). The loader
    # searches CWD before system dirs, so anything in this set is reachable.
    $localBins = @{}
    Get-ChildItem . -File -Include *.dll,*.exe,*.pyd -ErrorAction SilentlyContinue |
        ForEach-Object { $localBins[$_.Name.ToLower()] = $true }

    # System DLLs that ship with Windows / VC redist — always available, so
    # an import is not "missing" just because the file isn't in CWD.
    $systemDllPattern =
        '^(api-ms-win-|ext-ms-|kernel|user|gdi|advapi|ole|shell|comctl|comdlg|' +
        'ws2_|winmm|imm32|version|psapi|wininet|winhttp|crypt|secur|setupapi|' +
        'rpcrt4|netapi32|userenv|iphlpapi|dnsapi|wldap32|dbghelp|dbgeng|' +
        'msvcp|vcruntime|ucrtbase|msvcrt|concrt|mfc|d3d|dxgi|opengl|glu|' +
        'cudart|cublas|cufft|curand|cudnn|nvrtc|nv-cuda|' +
        'python3?\d?\.dll$|powrprof|cfgmgr32|bcrypt|ntdll|hid|propsys)'

    Write-Host ""
    Write-Host "=== Cross-check of all *.dll / *.exe / *.pyd imports against local CWD ==="
    Write-Host "    Flag with 'MISSING' = import target is not in CWD and not a known system DLL."
    Write-Host ""

    $binaries = Get-ChildItem . -File -Include *.dll,*.exe,*.pyd -ErrorAction SilentlyContinue |
                Sort-Object Name
    foreach ($bin in $binaries) {
        $imports = & $dumpbin /dependents $bin.FullName 2>$null |
                   Where-Object { $_ -match '\.dll$' } |
                   ForEach-Object { $_.Trim() }
        $missing = @()
        foreach ($imp in $imports) {
            $impLower = $imp.ToLower()
            if (-not $localBins.ContainsKey($impLower) -and $impLower -notmatch $systemDllPattern) {
                $missing += $imp
            }
        }
        if ($missing.Count -gt 0) {
            Write-Host "MISSING in $($bin.Name):"
            foreach ($m in $missing) { Write-Host "    -> $m" }
        }
    }

    # Keep the original per-target dumps so existing log readers still find them.
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
