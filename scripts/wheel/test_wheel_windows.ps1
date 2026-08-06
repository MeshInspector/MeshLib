param([Parameter(Mandatory)][string]$Version)

# Install the freshly-built meshlib wheel into the requested Python
# interpreter (via the `py` launcher) and run the wheel's pytest suite.
# Run from the repo root; the wheel artifact must already be unpacked in
# the working directory.

$ErrorActionPreference = 'Stop'

# base wheel plus the split-off meshlib_viewer wheel
$wheels = @(Get-ChildItem -Filter meshlib*win*.whl)
if ($wheels.Count -eq 0) { throw "No meshlib*win*.whl wheel found in $(Get-Location)" }

py -$Version -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
py -$Version -m pip uninstall -y meshlib
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
py -$Version -m pip install --upgrade -r ./requirements/python/requirements.txt
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
py -$Version -m pip install pytest
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
py -$Version -m pip install @($wheels | ForEach-Object FullName)
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Set-Location test_python
py -$Version -m pytest -s -v
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
