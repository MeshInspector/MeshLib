# Probe whether MSBuild can build with the requested MSVC platform toolsets on
# this runner. Having the MSVC compiler tools (VC\Tools\MSVC) is not enough --
# MSBuild also needs the platform toolset integration
# (PlatformToolsets\<vNNN>\Toolset.props); its absence is what raises MSB8020.
# Dumps the installed compiler tools and the platform toolsets MSBuild sees,
# then reports each requested toolset. Diagnostic only: it never fails.
[CmdletBinding()]
param(
    [string[]]$Toolsets = @('v142', 'v143')
)

$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
if (-not (Test-Path $vswhere)) {
    Write-Host "vswhere.exe not found at $vswhere; cannot check toolsets."
    return
}

$instances = & $vswhere -all -prerelease -products * -format json | ConvertFrom-Json
$found = @{}
foreach ($ts in $Toolsets) { $found[$ts] = @() }

foreach ($inst in $instances) {
    Write-Host "VS instance: $($inst.displayName) [$($inst.installationVersion)] @ $($inst.installationPath)"

    # Installed MSVC compiler tools -- context only, NOT what MSB8020 checks.
    $msvcDir = Join-Path $inst.installationPath 'VC\Tools\MSVC'
    if (Test-Path $msvcDir) {
        $vers = Get-ChildItem $msvcDir -Directory | Select-Object -ExpandProperty Name
        Write-Host "  MSVC compiler tools: $([string]::Join(', ', $vers))"
    }

    # Platform toolsets MSBuild sees -- this IS what MSB8020 checks.
    $vcRoot = Join-Path $inst.installationPath 'MSBuild\Microsoft\VC'
    foreach ($vcVer in (Get-ChildItem $vcRoot -Directory -ErrorAction SilentlyContinue)) {
        foreach ($plat in (Get-ChildItem (Join-Path $vcVer.FullName 'Platforms') -Directory -ErrorAction SilentlyContinue)) {
            $pts = Join-Path $plat.FullName 'PlatformToolsets'
            if (Test-Path $pts) {
                $names = Get-ChildItem $pts -Directory -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Name
                Write-Host "  PlatformToolsets [$($vcVer.Name)/$($plat.Name)]: $([string]::Join(', ', $names))"
                foreach ($ts in $Toolsets) {
                    $p = Join-Path $pts "$ts\Toolset.props"
                    if (Test-Path $p) { $found[$ts] += $p }
                }
            }
        }
    }
}

foreach ($ts in $Toolsets) {
    if ($found[$ts]) {
        Write-Host "$ts platform toolset: INSTALLED"
        $found[$ts] | ForEach-Object { Write-Host "  -> $_" }
    } else {
        Write-Host "$ts platform toolset: NOT INSTALLED (builds with PlatformToolset=$ts fail with MSB8020)."
    }
}
