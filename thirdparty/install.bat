set VCPKG_DEFAULT_TRIPLET=x64-windows-meshlib

for /f "delims=" %%i in ('where vcpkg') do set vcpkg_path=%%~dpi
if not exist "%vcpkg_path%downloads" mkdir "%vcpkg_path%downloads"
copy "%~dp0vcpkg\downloads\*" "%vcpkg_path%downloads"

setlocal enabledelayedexpansion
set packages=
for /f "delims=" %%i in ('type "%~dp0..\requirements\windows.txt"') do (
  set packages=!packages! %%i
)
vcpkg install !packages! --recurse --binarysource=clear --overlay-triplets "%~dp0vcpkg\triplets"
endlocal