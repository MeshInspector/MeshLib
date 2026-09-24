@echo off
setlocal enabledelayedexpansion

REM options:
REM   --write-s3                      push vcpkg binary cache to S3 (needs AWS credentials)
REM   --use-s3-asset-provider         fetch vcpkg download assets via thirdparty\asset-provider-s3.bat (S3 then curl).
REM                                   Use only when pinned to an older vcpkg whose upstream download URLs are stale or broken;
REM                                   newer vcpkg ports usually do not need this.
REM   --extra-requirements <file>     append packages from <file> to the install list (one package per line, vcpkg syntax).
REM                                   May be passed multiple times. Lets downstream callers append their own packages
REM                                   onto the same vcpkg invocation, so all the env-var and overlay setup lives here.

REM VCPKG_TAG is the S3 binary-cache folder, derived below from the checked-out
REM vcpkg release tag; it may not always exist in S3.
REM use "aws s3 ls s3://vcpkg-export/" to list all available tags

if not defined VCPKG_DEFAULT_TRIPLET set VCPKG_DEFAULT_TRIPLET=x64-windows-meshlib
echo Using vcpkg triplet: %VCPKG_DEFAULT_TRIPLET%

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
if "!write_s3_option!"=="true" (
    aws.exe --version >nul 2>&1
    if errorlevel 1 (
        echo "Error: --write-s3 requires AWS CLI to be installed."
        exit /b 1
    )
)

REM Detect vcpkg path
for /f "delims=" %%i in ('where vcpkg 2^>nul') do set vcpkg_path=%%~dpi
if not defined vcpkg_path (
    echo vcpkg not found. Setting VCPKG_TAG to "no-tag".
    set VCPKG_TAG=no-tag
) else (
    REM With --write-s3 a missing tag is fatal: pushing under "no-tag" would split the cache.
    set "VCPKG_TAG="
    for /f "delims=" %%T in ('git -c safe.directory^=* -C "!vcpkg_path!." describe --tags --exact-match 2^>nul') do set VCPKG_TAG=%%T
    if not defined VCPKG_TAG (
        if "!write_s3_option!"=="true" goto :tag_error
        set VCPKG_TAG=no-tag
    )
)

echo Using vcpkg version: !VCPKG_TAG!

REM Configure VCPKG_BINARY_SOURCES
REM Reading via HTTP is batched and thus usually faster than via x-aws
set "S3_REGION=us-east-1"
set "S3_URL=s3://vcpkg-export/!VCPKG_TAG!/!VCPKG_DEFAULT_TRIPLET!/"
set "HTTP_URL=https://vcpkg-export.s3.!S3_REGION!.amazonaws.com/!VCPKG_TAG!/!VCPKG_DEFAULT_TRIPLET!/{sha}.zip"
if "!write_s3_option!"=="true" (
    echo "Mode: pull-push vcpkg binary cache. AWS credentials are required."
    set "VCPKG_BINARY_SOURCES=clear;http,!HTTP_URL!,read;x-aws,!S3_URL!,write;"
) else (
    echo "Mode: pull vcpkg binary cache. No AWS credentials are required."
    set "VCPKG_BINARY_SOURCES=clear;http,!HTTP_URL!,read;"
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

for %%f in (!extra_req_files!) do (
    if not exist "%%~f" (
        echo Error: --extra-requirements file not found: %%~f
        exit /b 1
    )
    for /f "delims=" %%i in ('type "%%~f"') do (
        set packages=!packages! %%i
    )
)

REM On v142-pinned (VS2019) triplets, prepend ports-vs19 so its overlays win over the registry.
set "OVERLAY_PORTS_FLAGS="
if /I "%VCPKG_DEFAULT_TRIPLET%"=="x64-windows-vs2019-meshlib" set "OVERLAY_PORTS_FLAGS=--overlay-ports "%~dp0vcpkg\ports-vs19""
if /I "%VCPKG_DEFAULT_TRIPLET%"=="x64-windows-meshlib-iterator-debug" set "OVERLAY_PORTS_FLAGS=--overlay-ports "%~dp0vcpkg\ports-vs19""

REM Install vcpkg core dependencies
vcpkg install vcpkg-cmake vcpkg-cmake-config --host-triplet %VCPKG_DEFAULT_TRIPLET% --overlay-triplets "%~dp0vcpkg\triplets" --debug --x-abi-tools-use-exact-versions || goto :error

REM Install all required dependencies
vcpkg install !packages! --host-triplet %VCPKG_DEFAULT_TRIPLET% --overlay-triplets "%~dp0vcpkg\triplets" !OVERLAY_PORTS_FLAGS! --overlay-ports "%~dp0vcpkg\ports" --debug --x-abi-tools-use-exact-versions --recurse || goto :error

endlocal
goto :EOF

REM Error handling
:error
echo Failed with error #%errorlevel%.
endlocal
exit /b %errorlevel%

:tag_error
echo Error: could not determine the vcpkg release tag at "!vcpkg_path!":
git -c safe.directory=* -C "!vcpkg_path!." describe --tags --exact-match
endlocal
exit /b 1
