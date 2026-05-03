@echo off
setlocal enabledelayedexpansion

REM options:
REM   --write-s3                  push vcpkg binary cache to S3 (needs AWS credentials)
REM   --use-s3-asset-provider     fetch vcpkg download assets via thirdparty\asset-provider-s3.bat (S3 then curl).
REM                               Use only when pinned to an older vcpkg whose upstream download URLs are stale or broken;
REM                               newer vcpkg ports usually do not need this.
REM   --extra-requirements <file> append packages from <file> to the install list (one package per line, vcpkg syntax).
REM                               May be passed multiple times. Lets downstream repos (e.g. MeshInspectorCode) install
REM                               their extra packages in the same vcpkg invocation as MeshLib's, so all the env-var
REM                               and overlay setup lives in one place.

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
set "extra_req_files="
set "expect_extra_req=false"
for %%i in (%*) do (
    if "!expect_extra_req!"=="true" (
        set "extra_req_files=!extra_req_files! %%~i"
        set "expect_extra_req=false"
    ) else if /I "%%i"=="--write-s3" (
        set "write_s3_option=true"
    ) else if /I "%%i"=="--use-s3-asset-provider" (
        set "use_s3_assets=true"
    ) else if /I "%%i"=="--extra-requirements" (
        set "expect_extra_req=true"
    )
)
if "!expect_extra_req!"=="true" (
    echo Error: --extra-requirements requires a file path argument.
    exit /b 1
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

REM Append packages from any --extra-requirements files (downstream repos)
for %%f in (!extra_req_files!) do (
    if not exist "%%~f" (
        echo Error: --extra-requirements file not found: %%~f
        exit /b 1
    )
    echo Appending packages from %%~f
    for /f "delims=" %%i in ('type "%%~f"') do (
        set packages=!packages! %%i
    )
)

REM Install vcpkg core dependencies
vcpkg install vcpkg-cmake vcpkg-cmake-config --host-triplet %VCPKG_DEFAULT_TRIPLET% --overlay-triplets "%~dp0vcpkg\triplets" --debug --x-abi-tools-use-exact-versions || goto :error

REM Install all required dependencies
vcpkg install !packages! --host-triplet %VCPKG_DEFAULT_TRIPLET% --overlay-triplets "%~dp0vcpkg\triplets" --overlay-ports "%~dp0vcpkg\ports" --debug --x-abi-tools-use-exact-versions || goto :error

endlocal
goto :EOF

REM Error handling
:error
echo Failed with error #%errorlevel%.
endlocal
exit /b %errorlevel%
