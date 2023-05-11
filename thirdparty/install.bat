set VCPKG_DEFAULT_TRIPLET=x64-windows-meshlib

for /f "delims=" %%i in ('where vcpkg') do set vcpkg_path=%%i
if not exist %vcpkg_path%\..\downloads mkdir %vcpkg_path%\..\downloads
copy "%~dp0vcpkg\downloads" %vcpkg_path%\..\downloads

vcpkg install --recurse --binarysource=clear --overlay-triplets %~dp0vcpkg\triplets @"%~dp0\..\requirements\windows.txt"
