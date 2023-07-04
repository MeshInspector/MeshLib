set VCPKG_DEFAULT_TRIPLET=x64-windows-meshlib

for /f "delims=" %%i in ('where vcpkg') do set vcpkg_path=%%~dpi
if not exist "%vcpkg_path%downloads" mkdir "%vcpkg_path%downloads"
copy "%~dp0vcpkg\downloads\*" "%vcpkg_path%downloads"

setlocal enabledelayedexpansion
set packages=
for /f "delims=" %%i in ('type "%~dp0..\requirements\windows.txt"') do (
  set packages=!packages! %%i
)

REM Set the binary sources to use the S3 bucket
set VCPKG_BINARY_SOURCES=clear;x-aws,s3://vcpkg-export/2023.04.15/test/,readwrite;

vcpkg install !packages! --recurse --overlay-triplets "%~dp0vcpkg\triplets" --debug --x-abi-tools-use-exact-versions
endlocal
