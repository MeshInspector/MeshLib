# Welcome to the MeshLib!

[![build-test-distribute](https://github.com/MeshRUs/MeshLib/actions/workflows/build-test-distribute.yml/badge.svg?branch=master)](https://github.com/MeshRUs/MeshLib/actions/workflows/build-test-distribute.yml?branch=master)


## Build with VS2019 on Windows
```sh
$ git clone https://github.com/MeshRUs/MeshLib.git
$ cd MeshLib
$ git submodule update --init --recursive
```
### Preparing Third Parties
Some third parties are taken from vcpkg, while others (missing in vcpkg) are configured as git submodules.

### Vcpkg
1. Please install vcpkg, and integrate it into Visual Studio (note that vcpkg requires English laguage pack installed in Visual Studio):
    ```sh
    $ git clone https://github.com/Microsoft/vcpkg.git
    $ cd vcpkg
    $ git checkout 5c54cc06554e450829d72013c4b9e4baae41529a
    $ .\bootstrap-vcpkg.bat
    $ .\vcpkg integrate install (with admin rights)
    ```
    More details here: [vcpkg](https://github.com/microsoft/vcpkg).

2. Copy **thirdparty/vcpkg/triplets/x64-windows-meshrus.cmake** to **vcpkg/triplets** folder of vcpkg installation.
3. Execute install.bat
    ```sh
    $ cd vcpkg # or add vcpcg to PATH
    $ <path_to_MeshLib>/thirdparty/install.bat
    ```    
## Build with CMake on Linux
This installation was checked on Ubuntu 20.04.4.

Use automated installation process. It takes ~40 minutes if no required packages are already installed.
This approach is useful for new MR developers
**Install/Build dependencies. Build project. Run Test Application** Run the following in terminal:

    ```sh
    $ git clone https://github.com/MeshRUs/MeshLib.git
    $ cd MeshLib
    $ sudo ./scripts/build_thirdparty.sh # need sudo to check and install dependencies
	$ ./scripts/install_thirdparty.sh
    $ ./scripts/build_sources.sh
    $ ./scripts/distribution.sh
    $ sudo apt install ./distr/meshrus-dev.deb
    ```

Note! ./scripts/install*.sh scripts could be used as well, but apt install is prefferable.
Note! ./scripts/install*.sh scripts copy MR files directly to /usr/local/lib. Remove this directory manually if exists before apt install deb package
Note! You could specify build type to Debug by ```export MESHRUS_BUILD_TYPE=Debug```. Release is default.

