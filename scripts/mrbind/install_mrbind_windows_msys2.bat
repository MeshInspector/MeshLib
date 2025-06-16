@echo off

rem Builds the MRBind submodule at `MeshLib/mrbind/build`.
rem Before running this, run `install_deps_windows_msys2.bat`.

rem Push variables.
setlocal
rem Push enable extensions (for `mkdir` to behave like Linux `mkdir -p`).
setlocal enableextensions

if "%MSYS2_DIR%" == "" set MSYS2_DIR=C:\msys64_meshlib_mrbind

if "%MRBIND_DIR%" == "" set MRBIND_DIR=%~dp0\..\..\thirdparty\mrbind

rem Preserve the current directory. We'll do `popd` at the end...
pushd .

if not exist %MSYS2_DIR% (
    echo MSYS2 was NOT found at `%MSYS2_DIR%`. Run `install_deps_windows_msys2.bat` to install it.
) else (
    echo Found MSYS2 at `%MSYS2_DIR%`.

    cd %MRBIND_DIR%

    rem --- Build MRBind
    rmdir /S /Q build
    call %MSYS2_DIR%\msys2_shell.cmd -no-start -defterm -here -clang64 -c "cmake -B build -DCMAKE_BUILD_TYPE=RelWithDebugInfo -DCMAKE_EXPORT_COMPILE_COMMANDS=ON && cmake --build build"
)

rem Restore the original directory.
popd

rem Pop extensions.
endlocal
rem Pop variables.
endlocal
