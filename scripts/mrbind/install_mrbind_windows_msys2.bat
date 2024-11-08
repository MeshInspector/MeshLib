@echo off

rem Downloads the source code for MRBind and builds it, at `MeshLib/mrbind/build`.
rem Before running this, run `install_deps_windows_msys2.bat`.

rem Push variables, and enable extensions (for `mkdir` to behave like Linux `mkdir -p`).
setlocal enableextensions

if "%MSYS2_DIR%" == "" set MSYS2_DIR=C:\msys64_meshlib_mrbind

if "%MRBIND_DIR%" == "" set MRBIND_DIR=%~dp0\..\..\mrbind

if "%MRBIND_COMMIT%" == "" set /p MRBIND_COMMIT=<%~dp0\mrbind_commit.txt

rem Preserve the current directory. We'll do `popd` at the end...
pushd .

if not exist %MSYS2_DIR% (
    echo MSYS2 was NOT found at `%MSYS2_DIR%`. Run `install_deps_windows_msys2.bat` to build it.
) else (
    echo Found MSYS2 at `%MSYS2_DIR%`.


    rem --- Ensure the MRBind source exists. Pull the latest version if already exists.
    if exist %MRBIND_DIR% (
        echo Found MRBind sources at `%MRBIND_DIR%`.
        cd %MRBIND_DIR%
        git fetch
    ) else (
        echo Didn't find MRBind sources at `%MRBIND_DIR%`, cloning...

        rem Create the target directory first.
        mkdir %MRBIND_DIR%

        git clone https://github.com/MeshInspector/mrbind %MRBIND_DIR%
        cd %MRBIND_DIR%
    )

    git checkout %MRBIND_COMMIT%

    rem --- Build MRBind
    rmdir /S /Q build
    call %MSYS2_DIR%\msys2_shell.cmd -no-start -defterm -here -clang64 -c "cmake -B build -DCMAKE_BUILD_TYPE=RelWithDebugInfo && cmake --build build"
)

rem Restore the original directory.
popd

rem Pop variables.
endlocal
