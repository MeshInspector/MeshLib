param([Parameter(Mandatory)][string]$Version)

# Install the freshly-built meshlib wheel into the requested Python
# interpreter (via the `py` launcher) and run the wheel's pytest suite.
# Run from the repo root; the wheel artifact must already be unpacked in
# the working directory.

$ErrorActionPreference = 'Stop'

$wheel = Get-ChildItem -Filter meshlib*win*.whl
if ($null -eq $wheel) { throw "No meshlib*win*.whl wheel found in $(Get-Location)" }

py -$Version -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
py -$Version -m pip uninstall -y meshlib
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
py -$Version -m pip install --upgrade -r ./requirements/python/requirements.txt
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
py -$Version -m pip install pytest
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
py -$Version -m pip install $wheel
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Set-Location test_python
py -$Version -m pytest -s -v
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
