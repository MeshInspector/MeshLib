
REM choco install git.install sudo --confirm

set WORKDIR=%cd%
echo %cd%

if not exist C:\vcpkg\ (
  cd C:\\
  cd
  git clone https://github.com/Microsoft/vcpkg.git
)
cd C:\vcpkg\
git checkout 2022.11.14
call bootstrap-vcpkg.bat
sudo vcpkg integrate install

echo %WORKDIR%
cd %WORKDIR%

call thirdparty\install.bat
rmdir /S /Q %WORKDIR%\\vcpkg\\
robocopy C:\\vcpkg\\ %WORKDIR%\\vcpkg\\ *.* /XD "buildtrees" "downloads" "ports" "packages" /S