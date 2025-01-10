@echo off

rem Allow `!var!` variables, which is the only way you can set variables in loops.
setlocal enabledelayedexpansion
rem Allow `%RANDOM%`.
setlocal enableextensions

rem Download the page with the list of available versions. We need this to get the latest minor version number.
echo Downloading the list of known versions...
set "tempfile=%tmp%\meshlib_pythons_downloader_%RANDOM%.tmp"
del %tempfile% 2>nul
powershell -Command "Invoke-WebRequest https://www.python.org/ftp/python -OutFile %tempfile%"
echo Finished downloading to: %tempfile%

rem Extract the version numbers from the page and version-sort them. Write the result to `%tempfile%2`.
powershell (type %tempfile%) -match 'href=""""3\.' -replace '.*^>(.*)/^<.*','$1' "|" Sort-Object { $_ -as [version] } -Descending >%tempfile%2

for /f %%x in (%~dp0\python_versions.txt) do (
    rem Here %%x is the version number, such as 3.13.
    rem Set `!ver_terse!` to the version number without the dot, such as `313`.
    set ver_terse=%%x
    set ver_terse=!ver_terse:.=!
    if exist %localappdata%\Programs\Python\Python!ver_terse! (
        echo Python %%x - already installed
    ) else (
        echo Python %%x - installing...

        rem Guess the minor version number.
        set done=0
        for /f %%y in ('findstr %%x %tempfile%2') do (
            if !done! == 0 (
                echo Trying version: %%y
                set installer=%tmp%\python-%%y-amd64.exe
                powershell -Command "Invoke-WebRequest https://www.python.org/ftp/python/%%y/python-%%y-amd64.exe -OutFile !installer!" 2>nul
                if exist !installer! (
                    echo Download successful, installing... (click "yes" if prompted)
                    !installer! /quiet
                    set done=1
                ) else (
                    echo No Windows installer for this version.
                )
            )
        )
        if !done! == 0 (
            echo Couldn't find a suitable version to install.
            exit /b 1
        )
    )
)

echo Done

endlocal
endlocal