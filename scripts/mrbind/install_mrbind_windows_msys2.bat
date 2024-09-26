@echo off

rem Downloads the source code for MRBind and builds it, at `C:\msys64_meshlib_mrbind\home\username\mrbind\build`.
rem Before running this, run `install_deps_windows_msys2.bat`.

set MSYS2_DIR=C:\msys64_meshlib_mrbind

set MRBIND_DIR=%MSYS2_DIR%\home\%USERNAME%\mrbind

rem Preserve the current directory. We'll do `popd` at the end...
pushd .

if not exist %MSYS2_DIR% (
    echo MSYS2 was NOT found at `%MSYS2_DIR%`. Run `install_deps_windows_msys2.bat` to build it.
) else (
    echo Fount MSYS2 at `%MSYS2_DIR%`.


    rem --- Ensure the MRBind source exists. Pull the latest version if already exists.
    if exist %MRBIND_DIR% (
        echo Found MRBind sources at `%MRBIND_DIR%`. Pulling the latest version.
        cd %MRBIND_DIR%
        git checkout master
        git pull
    ) else (
        echo Didn't find MRBind sources at `%MRBIND_DIR%`, cloning...

        rem Create the target directory first.

        rem But first, make sure `mkdir` behaves as if with `-p`. This usually should be the default, but perhaps not on all systems.
        setlocal enableextensions
        mkdir %MRBIND_DIR%
        endlocal

        git clone https://github.com/MeshInspector/mrbind %MRBIND_DIR%
        cd %MRBIND_DIR%
    )

    rem --- Build MRBind
    rmdir /S /Q build
    call %MSYS2_DIR%\msys2_shell.cmd -no-start -defterm -here -clang64 -c "cmake -B build -DCMAKE_BUILD_TYPE=RelWithDebugInfo && cmake --build build"
)

rem Restore the original directory.
popd
