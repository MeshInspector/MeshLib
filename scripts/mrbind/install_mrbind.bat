@echo off

rem Automatically downloads the source for MRBind and builds it.
rem Before running this, run `install_msys2_tools.bat`.

set MSYS2_DIR=C:\msys64_meshlib_mrbind

set MRBIND_DIR=%MSYS2_DIR%\home\%USERNAME%\_mrbind

rem Preserve the current directory. We'll do `popd` at the end...
pushd .

if not exist %MSYS2_DIR% (
    echo MSYS2 was NOT found at `%MSYS2_DIR%`. Run `install_msys2_tools.bat` to build it.
) else (
    echo Fount MSYS2 at `%MSYS2_DIR%`.


    rem --- Ensure the MRBind source exists. Pull the latest version if already exists.
    if exist %MRBIND_DIR% (
        echo Found MRBind sources at `%MRBIND_DIR%`. Pulling the latest version.
        cd %MRBIND_DIR%
        git pull
    ) else (
        echo Didn't find MRBind sources at `%MRBIND_DIR%`, cloning...
        git clone https://github.com/MeshInspector/mrbind %MRBIND_DIR%
        cd %MRBIND_DIR%
    )

    rem --- Build MRBind
    rmdir /S /Q build
    call %MSYS2_DIR%\msys2_shell -no-start -defterm -here -clang64 -c "cmake -B build -DCMAKE_BUILD_TYPE=RelWithDebugInfo && cmake --build build"
)

rem Restore the original directory.
popd