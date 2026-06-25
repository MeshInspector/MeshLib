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

    # Index of binaries actually present in CWD (lower-cased). The Windows
    # loader searches the executable's directory first, so any import whose
    # filename is in this set is reachable.
    $localBins = @{}
    Get-ChildItem .\* -File -Include *.dll,*.exe,*.pyd -ErrorAction SilentlyContinue |
        ForEach-Object { $localBins[$_.Name.ToLower()] = $true }

    # System DLLs shipped by Windows / VC redist / Python / CUDA — always
    # findable via System32 or PATH, so don't flag them as missing.
    $systemDllPattern =
        '^(api-ms-win-|ext-ms-|kernel|user|gdi|advapi|ole|shell|comctl|comdlg|' +
        'ws2_|winmm|imm32|version|psapi|wininet|winhttp|crypt|secur|setupapi|' +
        'rpcrt4|netapi32|userenv|iphlpapi|dnsapi|wldap32|dbghelp|dbgeng|' +
        'msvcp|vcruntime|ucrtbase|msvcrt|concrt|mfc|d3d|dxgi|opengl|glu|' +
        'cudart|cublas|cufft|curand|cudnn|nvrtc|nv-cuda|' +
        'powrprof|cfgmgr32|bcrypt|ntdll|hid|propsys)'

    Write-Host ""
    Write-Host "=== Cross-check of all *.dll / *.exe / *.pyd imports against CWD ==="
    Write-Host "    MISSING = import target not in CWD and not a known system DLL."
    Write-Host ""

    $binaries = Get-ChildItem .\* -File -Include *.dll,*.exe,*.pyd -ErrorAction SilentlyContinue |
                Sort-Object Name
    Write-Host "    (scanning $($binaries.Count) binaries)"
    $missingTotal = 0
    foreach ($bin in $binaries) {
        $imports = & $dumpbin /dependents $bin.FullName 2>$null |
                   Where-Object { $_ -match '^\s+[A-Za-z0-9_+.-]+\.dll\s*$' } |
                   ForEach-Object { $_.Trim() }
        $missing = @()
        foreach ($imp in $imports) {
            $impLower = $imp.ToLower()
            if (-not $localBins.ContainsKey($impLower) -and $impLower -notmatch $systemDllPattern) {
                $missing += $imp
            }
        }
        if ($missing.Count -gt 0) {
            $missingTotal += $missing.Count
            Write-Host "MISSING in $($bin.Name):"
            foreach ($m in $missing) { Write-Host "    -> $m" }
        }
    }
    Write-Host ""
    Write-Host "    (cross-check complete: $missingTotal flagged imports)"

    # Per-target dumps for the binaries we most care about; keep the
    # original three plus the Python-embedding chain that runs in MRTest.exe
    # but not in MeshViewer.exe (so failures there are invisible to the
    # current Start-and-Exit smoke test).
    foreach ($target in 'MeshViewer.exe','MRMesh.dll','MRTest.exe',
                        'python312.dll','python3.dll',
                        'MREmbeddedPython.dll','MRPython.dll',
                        'pybind11nonlimitedapi_stubs.dll') {
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
