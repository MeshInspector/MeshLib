@echo off
setlocal
set "url=%~1"
set "sha512=%~2"
set "dst=%~3"

REM convert forward → backward slashes for Windows
set "_dst=%dst:/=\%"

REM vcpkg often spawns this script with a minimal PATH; interactive shells still find aws.
set "AWS_EXE="
if exist "%ProgramFiles%\Amazon\AWSCLIV2\aws.exe" set "AWS_EXE=%ProgramFiles%\Amazon\AWSCLIV2\aws.exe"
if not defined AWS_EXE if exist "%ProgramFiles(x86)%\Amazon\AWSCLIV2\aws.exe" set "AWS_EXE=%ProgramFiles(x86)%\Amazon\AWSCLIV2\aws.exe"
if not defined AWS_EXE set "AWS_EXE=aws"

REM extract basename from URL (last path segment; unquoted %%i matches prior behavior)
for %%i in (%url%) do set "basename=%%~nxi"

echo S3 key: %basename%
echo dst: %dst%
echo dst: %_dst%
echo S3 cmd: "%AWS_EXE%" s3 cp "s3://vcpkg-export/downloads/%basename%" "%_dst%" --no-sign-request

"%AWS_EXE%" s3 cp "s3://vcpkg-export/downloads/%basename%" "%_dst%" --no-sign-request
if %errorlevel% equ 0 if exist "%_dst%" (
    echo Fetched from S3: %basename%
    exit /b 0
)

echo Fallback to original URL.
curl.exe -L -f -s -S -o "%_dst%" "%url%"
exit /b %errorlevel%
