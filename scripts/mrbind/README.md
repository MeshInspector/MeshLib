## Generating the bindings

### Windows:

Run those once:

* `scripts/mrbind/install_msys2_tools.bat` to install MSYS2 with Clang and libclang. Re-run to update to the latest version.

* `scripts/mrbind/install_mrbind.bat` to download and compile MRBind. Re-run to update to the latest version.

Select Visual Studio installation:

* Open the Developer Command prompt for the desired version of VS.

* Run `echo %VCToolsInstallDir%`. Copy the printed value.

* Prepend `VCToolsInstallDir='...'` to the build command.

  E.g. `VCToolsInstallDir='C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.29.30133\'`