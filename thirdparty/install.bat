@echo off
setlocal enabledelayedexpansion

REM options:
REM   --write-s3                  push vcpkg binary cache to S3 (needs AWS credentials)
REM   --use-s3-asset-provider     fetch vcpkg download assets via thirdparty\asset-provider-s3.bat (S3 then curl).
REM                               Use only when pinned to an older vcpkg whose upstream download URLs are stale or broken;
REM                               newer vcpkg ports usually do not need this.

REM The VCPKG_TAG variable represents the S3 folder and may not always exist in S3
REM use "aws s3 ls s3://vcpkg-export/" to list all available tags

if not defined VCPKG_DEFAULT_TRIPLET set VCPKG_DEFAULT_TRIPLET=x64-windows-meshlib
echo Using vcpkg triplet: %VCPKG_DEFAULT_TRIPLET%

REM Check if AWS CLI is installed
set "aws_cli_available=false"
aws.exe --version >nul 2>&1
if errorlevel 1 (
    echo AWS CLI v2: not found
    echo "Without AWS CLI, vcpkg cache from S3 will not be available, and dependencies will be built from source"
) else (
    echo AWS CLI v2: found
    echo "Vcpkg binary cache (if available) will be downloaded from S3"
    set "aws_cli_available=true"
)

REM Detect vcpkg path
for /f "delims=" %%i in ('where vcpkg 2^>nul') do set vcpkg_path=%%~dpi
if not defined vcpkg_path (
    echo vcpkg not found. Setting VCPKG_TAG to "no-tag".
    set VCPKG_TAG=no-tag
) else (
    REM Extract version number (YYYY-MM-DD-hash) and cut first 10 characters
    for /f "tokens=6" %%V in ('vcpkg version 2^>nul ^| findstr /R "vcpkg package management program version [0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]"') do set FULL_VCPKG_TAG=%%V
    set VCPKG_TAG=!FULL_VCPKG_TAG:~0,10!
)

echo Using vcpkg version: !VCPKG_TAG!

REM Check for CLI options
set "write_s3_option=false"
set "use_s3_assets=false"
for %%i in (%*) do (
    if /I "%%i"=="--write-s3" set "write_s3_option=true"
    if /I "%%i"=="--use-s3-asset-provider" set "use_s3_assets=true"
)
if "!write_s3_option!"=="true" if "!aws_cli_available!"=="false" (
    echo "Error: --write-s3 requires AWS CLI to be installed."
    exit /b 1
)

REM Configure VCPKG_BINARY_SOURCES (only use s3 cache when aws cli is available)
if "!aws_cli_available!"=="true" (
    if "!write_s3_option!"=="true" (
        echo "Mode: pull-push vcpkg binary cache. AWS credentials are required."
        set "VCPKG_BINARY_SOURCES=clear;x-aws,s3://vcpkg-export/!VCPKG_TAG!/!VCPKG_DEFAULT_TRIPLET!/,readwrite;"
    ) else (
        echo "Mode: pull vcpkg binary cache. No AWS credentials are required."
        set "VCPKG_BINARY_SOURCES=clear;x-aws-config,no-sign-request;x-aws,s3://vcpkg-export/!VCPKG_TAG!/!VCPKG_DEFAULT_TRIPLET!/,readwrite;"
    )
) else (
    echo "Mode: build from source (no S3 binary cache)."
    set "VCPKG_BINARY_SOURCES=clear"
)

if "!use_s3_assets!"=="true" (
    echo Mode: S3 asset provider ^(thirdparty\asset-provider-s3.bat^).
    set "SCRIPT_PATH=%~dp0asset-provider-s3.bat"
    set "X_VCPKG_ASSET_SOURCES=clear;x-script,!SCRIPT_PATH! {url} {sha512} {dst}"
) else (
    set "X_VCPKG_ASSET_SOURCES="
)

REM Ensure vcpkg downloads folder exists
if not exist "!vcpkg_path!downloads\" mkdir "!vcpkg_path!downloads"
if exist "%~dp0vcpkg\downloads\" (
    xcopy "%~dp0vcpkg\downloads\*" "!vcpkg_path!downloads" /Y /E 2>nul
)

REM Read package list from requirements file
set packages=
for /f "delims=" %%i in ('type "%~dp0..\requirements\windows.txt"') do (
    set packages=!packages! %%i
)

REM Build the list of overlay-port flags. The VS2019/2024.10.21 build adds
REM ports-vs19 in front so its boost-interprocess (with the iterator-
REM invalidation patch from boostorg/interprocess#224) wins over the
REM upstream port. Other builds use only the shared overlay dir.
set "OVERLAY_PORTS_FLAGS=--overlay-ports "%~dp0vcpkg\ports""
if /I "%MESHLIB_VS2019_OVERLAY%"=="true" (
    set "OVERLAY_PORTS_FLAGS=--overlay-ports "%~dp0vcpkg\ports-vs19" --overlay-ports "%~dp0vcpkg\ports""
)

REM Install vcpkg core dependencies
vcpkg install vcpkg-cmake vcpkg-cmake-config --host-triplet %VCPKG_DEFAULT_TRIPLET% --overlay-triplets "%~dp0vcpkg\triplets" --debug --x-abi-tools-use-exact-versions || goto :error

REM Install all required dependencies
vcpkg install !packages! --host-triplet %VCPKG_DEFAULT_TRIPLET% --overlay-triplets "%~dp0vcpkg\triplets" !OVERLAY_PORTS_FLAGS! --debug --x-abi-tools-use-exact-versions || goto :error

endlocal
goto :EOF

REM Error handling
:error
echo Failed with error #%errorlevel%.
endlocal
exit /b %errorlevel%
