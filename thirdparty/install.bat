set VCPKG_DEFAULT_TRIPLET=x64-windows-meshrus

for /f "delims=" %%i in ('where vcpkg') do set vcpkg_path=%%i
if not exist %vcpkg_path%\..\downloads mkdir %vcpkg_path%\..\downloads
copy "%~dp0vcpkg\downloads" %vcpkg_path%\..\downloads

vcpkg install "@%~dp0\..\requirements\windows.txt" --recurse --binarysource=clear
