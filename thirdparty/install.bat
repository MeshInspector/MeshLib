@echo off
REM The VCPKG_TAG variable represents the S3 folder and may not always exist in S3
REM use "aws s3 ls s3://vcpkg-export/" to ls all available tags
set VCPKG_DEFAULT_TRIPLET=x64-windows-meshlib
setlocal enabledelayedexpansion


REM Default settings
set VCPKG_TAG=no-tag
set VCPKG_DEFAULT_TRIPLET=x64-windows-meshlib

REM Check if AWS CLI v2 is installed
REM AWS CLI is required for using vcpkg binary caching with S3.
aws --version >nul 2>&1
if errorlevel 1 (
    echo AWS CLI v2 is not installed.
    echo Without AWS CLI, vcpkg cache from S3 will not be available, and dependencies will be built from source.
) else (
    echo AWS CLI v2 is installed.
    echo vcpkg can use S3 for binary caching, reducing build times.
)

REM Detect vcpkg path
for /f "delims=" %%i in ('where vcpkg 2^>nul') do set vcpkg_path=%%~dpi
if not defined vcpkg_path (
    echo vcpkg not found. Setting VCPKG_TAG to "no-tag".
    set VCPKG_TAG=no-tag
) else (
    REM Extract the first valid version number (YYYY-MM-DD format)
    for /f "tokens=4 delims= " %%V in ('vcpkg version 2^>nul ^| findstr /R "[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]"') do set VCPKG_TAG=%%V
)

REM Check for --write-s3 option
set "write_s3_option=false"
for %%i in (%*) do (
    if /I "%%i"=="--write-s3" set "write_s3_option=true"
)

echo Using vcpkg version: %VCPKG_TAG%

REM Configure VCPKG_BINARY_SOURCES
if "!write_s3_option!"=="true" (
    set "VCPKG_BINARY_SOURCES=clear;x-aws,s3://vcpkg-export/%VCPKG_TAG%/x64-windows-meshlib/,readwrite;"
    echo "Using AWS authentication"
) else (
    set "VCPKG_BINARY_SOURCES=clear;x-aws-config,no-sign-request;x-aws,s3://vcpkg-export/%VCPKG_TAG%/x64-windows-meshlib/,readwrite;"
    echo "Using no authentication"
)

REM Ensure vcpkg downloads folder exists
if not exist "%vcpkg_path%downloads" mkdir "%vcpkg_path%downloads"
copy "%~dp0vcpkg\downloads\*" "%vcpkg_path%downloads" 2>nul

REM Read package list from requirements file
set packages=
for /f "delims=" %%i in ('type "%~dp0..\requirements\windows.txt"') do (
    set packages=!packages! %%i
)

REM Install dependencies
vcpkg install vcpkg-cmake vcpkg-cmake-config --host-triplet %VCPKG_DEFAULT_TRIPLET% --overlay-triplets "%~dp0vcpkg\triplets" --debug --x-abi-tools-use-exact-versions || goto :error
vcpkg install !packages! --host-triplet %VCPKG_DEFAULT_TRIPLET% --overlay-triplets "%~dp0vcpkg\triplets" --debug --x-abi-tools-use-exact-versions || goto :error

endlocal
goto :EOF

REM Error handling
:error
echo Failed with error #%errorlevel%.
endlocal
exit /b %errorlevel%
