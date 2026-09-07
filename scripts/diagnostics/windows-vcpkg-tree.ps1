# Inventory the vcpkg installed tree so DLL-not-found failures later in the
# build can be traced back to whether the DLL was (a) never installed,
# (b) installed under an unexpected name, or (c) installed fine but not
# copied to the build output.
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Triplet
)

Write-Host "=== Triplet: $Triplet ==="
foreach ($sub in 'bin', 'debug\bin', 'lib', 'debug\lib') {
    $dir = "C:\vcpkg\installed\$Triplet\$sub"
    Write-Host ""
    Write-Host "--- $dir ---"
    if (Test-Path $dir) {
        Get-ChildItem $dir -File -ErrorAction SilentlyContinue |
            Select-Object Name, @{N='SizeKB';E={[int]($_.Length/1KB)}} |
            Sort-Object Name | Format-Table -AutoSize
    } else {
        Write-Host "(directory not present)"
    }
}
