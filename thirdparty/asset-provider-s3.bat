@echo off
setlocal enabledelayedexpansion
set "url=%~1"
set "sha512=%~2"

REM Destination path: normally %3. If the URL was NOT quoted, cmd splits ?full_index=1 into
REM extra tokens and %3 becomes the hash — the real path is then %4. Pick first arg that has \.
set "dst=%~3"
echo.%~3| findstr "\\" >nul 2>&1 || set "dst=%~4"
if not defined dst set "dst=%~3"

REM convert forward → backward slashes for Windows
set "_dst=%dst:/=\%"

REM filename + extension only (e.g. openssl-openssl-....patch.6124.part)
for %%F in ("%_dst%") do set "DST_ARG=%%~nxF"


REM vcpkg often spawns this script with a minimal PATH; interactive shells still find aws
set "AWS_EXE="
if exist "%ProgramFiles%\Amazon\AWSCLIV2\aws.exe" set "AWS_EXE=%ProgramFiles%\Amazon\AWSCLIV2\aws.exe"
if not defined AWS_EXE if exist "%ProgramFiles(x86)%\Amazon\AWSCLIV2\aws.exe" set "AWS_EXE=%ProgramFiles(x86)%\Amazon\AWSCLIV2\aws.exe"
if not defined AWS_EXE set "AWS_EXE=aws"

set "fn=!DST_ARG!"
if /I "!fn:~-5!"==".part" set "fn=!fn:~0,-5!"
set "DST_KEY="
if defined fn for %%a in ("!fn!") do set "DST_KEY=%%~na"

REM S3 object key = URL basename (last path segment)
for %%i in (%url%) do set "basename=%%~nxi"

echo url: %url%
echo dst: %_dst%

echo S3 key1: !DST_KEY!
echo S3 key2: %basename%
echo S3 cmd1: "%AWS_EXE%" s3 cp "s3://vcpkg-export/downloads/!DST_KEY!" "%_dst%" --no-sign-request
echo S3 cmd2: "%AWS_EXE%" s3 cp "s3://vcpkg-export/downloads/%basename%" "%_dst%" --no-sign-request

if defined DST_KEY (
    "%AWS_EXE%" s3 cp "s3://vcpkg-export/downloads/!DST_KEY!" "%_dst%" --no-sign-request
    if not errorlevel 1 if exist "%_dst%" (
        echo Fetched from S3 ^(key1 / dst^): !DST_KEY!
        exit /b 0
    )
)

if defined basename if /I not "!basename!"=="!DST_KEY!" (
    "%AWS_EXE%" s3 cp "s3://vcpkg-export/downloads/%basename%" "%_dst%" --no-sign-request
    if not errorlevel 1 if exist "%_dst%" (
        echo Fetched from S3 ^(key2 / url^): %basename%
        exit /b 0
    )
)

echo S3 download failed.
exit /b 1
