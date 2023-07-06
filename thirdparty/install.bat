set VCPKG_DEFAULT_TRIPLET=x64-windows-meshlib

REM check if we could use aws s3 binary caching
aws --version >nul 2>&1
if errorlevel 1 (
    echo AWS CLI v2 is not installed.
) else (
    echo AWS CLI v2 is installed.
    set "VCPKG_BINARY_SOURCES=clear;x-aws,s3://vcpkg-export/2023.04.15/x64-windows-meshlib/,readwrite;"
)

for /f "delims=" %%i in ('where vcpkg') do set vcpkg_path=%%~dpi
if not exist "%vcpkg_path%downloads" mkdir "%vcpkg_path%downloads"
copy "%~dp0vcpkg\downloads\*" "%vcpkg_path%downloads"

setlocal enabledelayedexpansion
set packages=
for /f "delims=" %%i in ('type "%~dp0..\requirements\windows.txt"') do (
  set packages=!packages! %%i
)

vcpkg install vcpkg-cmake vcpkg-cmake-config --host-triplet x64-windows-meshlib --overlay-triplets "%~dp0vcpkg\triplets" --x-abi-tools-use-exact-versions
vcpkg install !packages! --host-triplet x64-windows-meshlib --overlay-triplets "%~dp0vcpkg\triplets" --x-abi-tools-use-exact-versions

endlocal