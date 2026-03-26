@echo off

rem This script installs (or updates) MSYS2, and some Clang tools in it.
rem Which is everything you need to generate MRBind bindings on Windows.

rem We create a separate copy of MSYS2 at C:\msys64_meshlib_mrbind for simplicity.
rem If you don't like that, you can manually install all dependencies into
rem some other MSYS2 copy.

setlocal

rem Make sure we're not running from the VS dev prompt.
rem Doing this just in case. `scripts/mrbind/install_mrbind_windows_msys2.bat` is known not to work there,
rem   but this script could possibly work. But it's easier to not support this.
if not "%VCToolsInstallDir%" == "" (
    echo Must not run this script from the VS developer command prompt. Use the regular terminal.
    exit /b 1
)

if "%MSYS2_DIR%" == "" set MSYS2_DIR=C:\msys64_meshlib_mrbind

rem ------ Ensure MSYS2 is installed

if exist %MSYS2_DIR% (
    echo MSYS2 is already installed to `%MSYS2_DIR%`.
) else (
    echo MSYS2 was NOT found at `%MSYS2_DIR%`.

    rem --- Download installer
    if exist %TMP%\msys2-base-x86_64-latest.sfx.exe (
        echo Will use existing installer at `%TMP%\msys2-base-x86_64-latest.sfx.exe`.
        echo     If it misbehaves, delete it and restart this script to download a new one. Or install manually to `%MSYS2_DIR%` and restart this script to download the required packages into it.
    ) else (
        echo Downloading installer to `%TMP%\msys2-base-x86_64-latest.sfx.exe`...
        rem This seems to automatically overwrite the file!
        powershell -Command "Invoke-WebRequest https://github.com/msys2/msys2-installer/releases/latest/download/msys2-base-x86_64-latest.sfx.exe -OutFile %TMP%\msys2-base-x86_64-latest.sfx.exe.part"
        move /Y %TMP%\msys2-base-x86_64-latest.sfx.exe.part %TMP%\msys2-base-x86_64-latest.sfx.exe
    )

    rem --- Run the installer
    echo Installing MSYS2...
    %TMP%\msys2-base-x86_64-latest.sfx.exe -y -o%MSYS2_DIR%
    rem Move files out of the `msys64` subdirectory.
    for %%a in ("%MSYS2_DIR%\msys64\*") do move /y "%%~fa" %MSYS2_DIR%
    for /d %%a in ("%MSYS2_DIR%\msys64\*") do move /y "%%~fa" %MSYS2_DIR%
    rmdir msys64
)

rem ------ Install MSYS2 packages
call %MSYS2_DIR%\msys2_shell.cmd -no-start -defterm -here -c "'%~dp0\msys2_download_packages.sh' && '%~dp0\msys2_install_packages.sh'"
echo Please ignore the errors above, if any, after the words `:: Running post-transaction hooks...`.

endlocal
