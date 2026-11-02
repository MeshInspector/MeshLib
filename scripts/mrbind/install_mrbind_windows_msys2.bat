@echo off

rem Builds the MRBind submodule at `MeshLib/mrbind/build`.

rem Push variables.
setlocal
rem Push enable extensions (for `mkdir` to behave like Linux `mkdir -p`).
setlocal enableextensions

rem Make sure we're not running from the VS dev prompt. It's known to break MRBind builds,
rem   because it somehow forces MSYS2 CMake to use Vcpkg. Possibly via an env variable, I didn't investiate further.
if not "%VCToolsInstallDir%" == "" (
    echo Must not run this script from the VS developer command prompt. Use the regular terminal.
    exit /b 1
)

if "%MSYS2_DIR%" == "" set MSYS2_DIR=C:\msys64_meshlib_mrbind

rem MSYS2 environment to build in: clang64 targets x86_64, clangarm64 targets aarch64.
if "%MSYS2_ENV%" == "" set MSYS2_ENV=clang64

if "%MRBIND_DIR%" == "" set MRBIND_DIR=%~dp0\..\..\thirdparty\mrbind

rem Preserve the current directory. We'll do `popd` at the end...
pushd .

if not exist %MSYS2_DIR% (
    echo MSYS2 was NOT found at `%MSYS2_DIR%`. Running `install_deps_windows_msys2.bat` to install it.
    call "%~dp0\install_deps_windows_msys2.bat"
    if errorlevel 1 (
        echo `install_deps_windows_msys2.bat` failed.
        popd
        exit /b 1
    )
)

echo Found MSYS2 at `%MSYS2_DIR%`.

cd %MRBIND_DIR%

rem --- Build MRBind
rem With `MRBIND_GENERATORS_ONLY=1`, only the generators working from the C bindings description: no parser and no `mrbind_gen_c`.
set MRBIND_CONFIGURE_ARGS=
set MRBIND_BUILD_ARGS=
if "%MRBIND_GENERATORS_ONLY%" == "1" (
    set MRBIND_CONFIGURE_ARGS=-DMRBIND_BUILD_PARSER=OFF
    set MRBIND_BUILD_ARGS=--target mrbind_gen_csharp mrbind_gen_rust mrbind_gen_java mrbind_gen_python
)
rmdir /S /Q build
call %MSYS2_DIR%\msys2_shell.cmd -no-start -defterm -here -%MSYS2_ENV% -c "cmake -B build -DCMAKE_BUILD_TYPE=RelWithDebugInfo -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo %MRBIND_CONFIGURE_ARGS% && cmake --build build %MRBIND_BUILD_ARGS%"

rem Restore the original directory.
popd

rem Pop extensions.
endlocal
rem Pop variables.
endlocal
