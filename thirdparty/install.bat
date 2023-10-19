@echo off
set VCPKG_DEFAULT_TRIPLET=x64-windows-meshlib

setlocal enabledelayedexpansion
REM check if we could use aws s3 binary caching
aws --version >nul 2>&1


for /f "delims=" %%i in ('where vcpkg') do set vcpkg_path=%%~dpi
if not exist "%vcpkg_path%downloads" mkdir "%vcpkg_path%downloads"
copy "%~dp0vcpkg\downloads\*" "%vcpkg_path%downloads"

set packages=
for /f "delims=" %%i in ('type "%~dp0..\requirements\windows.txt"') do (
  set packages=!packages! %%i
)

vcpkg install vcpkg-cmake vcpkg-cmake-config --host-triplet x64-windows-meshlib --overlay-triplets "%~dp0vcpkg\triplets"  --debug --x-abi-tools-use-exact-versions
vcpkg install !packages! --host-triplet x64-windows-meshlib --overlay-triplets "%~dp0vcpkg\triplets" --debug --x-abi-tools-use-exact-versions

endlocal
