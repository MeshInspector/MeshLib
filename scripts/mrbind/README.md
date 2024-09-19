# Generating the bindings

## Installing prerequisites

Run **`scripts/mrbind/install_deps_<platform>`** to install the dependencies (on Linux and MacOS - as root), then **`scripts/mrbind/install_mrbind_<platform>`** to build MRBind (not at root).

You can re-run those scripts to update the dependencies and/or MRBind itself.

There's no rocket science in those, and you can do it manually instead of running the scripts (especially building MRBind yourself), see below.

Among other things, the scripts can do following:

* On Linux and MacOS, create `~/mrbind` to build MRBind in.

* On Ubuntu, add [the LLVM repository](https://apt.llvm.org/) to install the latest Clang and libclang from.

* On Windows, install MSYS2 to `C:\msys64_meshlib_mrbind`.

* On Ubuntu 20.04, build GNU Make from source and install it to `/usr/local`, because the default one is outdated.

**More details and manual instructions:**

<details><summary><b>Windows</b></summary>

* **Installing dependencies:**

    On Windows we use MSYS2, because it provides prebuilt libclang and provides GNU Make to run our makefile.

    MSYS2 is a package manager, roughly speaking. They provide a bunch of MinGW-related packages (compilers and prebuilt libraries). Luckily Clang can always cross-compile, so MSYS2's MinGW Clang can produce MSVC-compatible executables with the correct flags. You still need to have VS installed though, since it will use its libraries.

    We use [MSYS2 CLANG64](https://www.msys2.org/docs/environments/) environment. Consult `install_deps_windows_msys2.bat` for the list of packages we install in it.

    Notably on Windows we don't have control over the Clang version, since MSYS2 only supports installing the latest one.

* **Building MRBind:**

    MRBind source code is at https://github.com/MeshInspector/mrbind/.

    We build MRBind at `C:\msys64_meshlib_mrbind\home\username\mrbind`, but you can build it elsewhere manually.

    We build in [MSYS2 CLANG64](https://www.msys2.org/docs/environments/) environment, using MSYS2's Clang. Other compilers are not guaranteed to work.


</details>

<details><summary><b>Ubuntu</b></summary>

* **Installing dependencies:**

    We want a certain version of Clang (see `preferred_clang_version.txt`), and since older versions of Ubuntu don't have it, we add Clang's repository: https://apt.llvm.org

    Also we need a certain version of GNU Make, so on Ubuntu 20.04 we build it from source and install to `/usr/local`.

    And obviously we install some packages, see `install_deps_ubuntu.sh` for the list.

* **Building MRBind:**

    MRBind source code is at https://github.com/MeshInspector/mrbind/.

    We build MRBind at `~/mrbind`, but you can build it elsewhere manually.

    You might want to pass `-DClang_DIR=/usr/lib/cmake/clang-VERSION` (where `VERSION` is the one mentioned in `preferred_clang_version.txt`) if you have several versions of libclang installed, because otherwise CMake might pick an arbitrary one (apparently it picks the first one returned by globbing `clang-*`, which might not be the latest one).

    Use `CC=clang-VERSION CXX=clang++-VERSION cmake ....` to build using Clang. Other compilers might work, but that's not guaranteed.

</details>

<details><summary><b>MacOS</b></summary>

* **Installing dependencies:**

    Homebrew must already be installed.

    We install a certain version of Clang and libclang from it (see `preferred_clang_version.txt`), and also GNU Make and Gawk. MacOS has its own Make, but it's outdated. It seems to have Gawk, but we install our own just in case.

    What we install from Brew is the regular Clang, not Apple Clang (Apple's fork for Clang), because that is based on an outdated branch of Clang.

    You must run following to add the installed things to your PATH:
    ```sh
    export PATH="/opt/homebrew/opt/make/libexec/gnubin:$PATH"
    export PATH="/opt/homebrew/opt/llvm/bin@VERSION:$PATH" # See the correct VERSION in `preferred_clang_version.txt`.
    ```

* **Building MRBind:**

    MRBind source code is at https://github.com/MeshInspector/mrbind/.

    We build MRBind at `~/mrbind`, but you can build it elsewhere manually.

    Make sure your PATH is correct, as explained in the previous step.

    Use `CC=clang-VERSION CXX=clang++-VERSION cmake ....` to build using the Clang we've installed. It might build using Apple Clang as well (if you don't set your PATH as explained above), but that's not guaranteed to work. (If you want to try it, you must pass `-DCMAKE_PREFIX_PATH=/opt/homebrew/opt/llvm` for it to find our libclang.)

    You might want to pass `-DClang_DIR=/usr/lib/cmake/clang-VERSION` (where `VERSION` is the one mentioned in `preferred_clang_version.txt`) if you have several versions of libclang installed, because otherwise CMake might pick an arbitrary one (apparently it picks the first one returned by globbing `clang-*`, which might not be the latest one).


</details>

### MacOS prerequisites:

* **Install some helper tools:**
  ```sh
  brew install make gawk
  ```
  If already installed, only run the second line to add it to PATH.

* **Install LLVM/Clang:**
  ```sh
  brew install llvm

  ```
  If already installed, only run the second line to add it to PATH.

* **Build MRBind:**
  ```sh
  git clone https://github.com/MeshInspector/mrbind ~/mrbind
  cd ~/mrbind
  CC=clang CXX=clang++ cmake -B build
  cmake --build build
  ```
  You must include homebrew's Clang in PATH as was expained above.

  Also, while it doesn't seem to be necessary if you build with homebrew Clang, you can pass  to specify the location of libclang (e.g. if you build with Apple Clang rather than homebrew Clang?).

### Linux prerequisites:

* **Install Clang 18 and libclang:**

  **On Ubuntu 24.04**, install directly from official repos:
  ```
  VER=18
  sudo apt install clang-$VER lld-$VER clang-tools-$VER libclang-$VER-dev llvm-$VER-dev
  ```
  **On older Ubuntu** this version is not in the official repos, must install from LLVM ones:
  ```
  VER=18
  wget https://apt.llvm.org/llvm.sh
  chmod +x llvm.sh
  sudo ./llvm.sh $VER
  rm llvm.sh
  sudo apt install clang-$VER lld-$VER clang-tools-$VER libclang-$VER-dev llvm-$VER-dev
  ```

* **Build MRBind:**
  ```sh
  git clone https://github.com/MeshInspector/mrbind ~/mrbind
  cd ~/mrbind
  CC=clang CXX=clang++ cmake -B build -DClang_DIR=/usr/lib/cmake/clang-18
  cmake --build build
  ```
  `Clang_DIR` is needed if you have more than one version of libclang installed, otherwise CMake might pick the wrong one (not even necessarily the newest one).


### Creating the bindings:

You need to run:
* On Windows: `scripts/mrbind/generate_win.bat ....` from VS developer command prompt (x64).
* On Linux and MacOS: `make -f scripts/mrbind/generate.mk ....`

Here are some example configurations, but read the docs below for details.

Note that we build in release mode by default, pass `MODE=none` or `MODE=debug` for faster build times.

* **Windows:** `scripts/mrbind/generate_win.bat -B --trace VS_MODE=Release` (or `=Debug`)

  The current directory matters, will look for MeshLib build in `./source/x64/$VS_MODE`.

* **MacOS:** `make -f scripts/mrbind/generate.mk -B --trace CXX='clang++ -fclang-abi-compat=17' MRBIND_SOURCE=/Users/user/repos/mrbind MESHLIB_SHLIB_DIR=build/RelWithDebugInfo/bin PYTHON_PKGCONF_NAME=python-3.10-embed`

  The current directory affects where the temprary files will be stored (in `./build`).

* **Ubuntu:** `make -f scripts/mrbind/generate.mk -B --trace CXX='clang++-18 -fclang-abi-compat=17' MRBIND_SOURCE=/home/user/repos/mrbind MESHLIB_SHLIB_DIR=build/RelWithDebugInfo/bin` (drop `-fclang-abi-compat=17` on Ubuntu 24.04 and newer).

  The current directory affects where the temprary files will be stored (in `./build`).

#### Selecting MeshLib build

On Windows:

* The current directory matters. We look for MeshLib in `./source/x64/$VS_MODE` relative to the current directory.

  By default `VS_MODE=Debug`. You can pass `VS_MODE=Release` as a flag to Make (note, not as an env variable).

On Linux and MacOS:

* Must pass `MESHLIB_SHLIB_DIR=...` (as a Make flag, not env variable) pointing to your build `bin` directory, e.g. `build/RelWithDebugInfo/bin`.

  The current directory only affects where the temporary files will be stored.

#### Compiler settings

By default we use `MODE=release`, which enables optimization and disables debug symbols.

Pass `MODE=none` for faster builds (but without debug symbols), or `MODE=debug` for builds with debug symbols.

Alternatively you can set `EXTRA_CFLAGS=... EXTRA_LDFLAGS=...` for fully custom compiler and linker flags.

#### Full vs incremental build

`-B` forces a full rebuild. Removing this flag in theory performs an incremental build, but changing anything in MeshLib headers requires a full rebuild anyway. Removing this flag is only useful for rebuilding auxiliary parts of the module (copying `__init__.py` again, and so on).

#### Selecting MRBind installation

On Windows, if you used `install_mrbind_windows_msys2.bat`, you don't need to worry about this.

On Other platforms, you must pass `MRBIND_SOURCE=path/to/mrbind` to Make (note, not as an env variable), pointing to the MRBind source directory.

If MRBind is not at `$MRBIND_SOURCE/build/mrbind`, must also pass `MRBIND_EXE` pointing to the executable.

#### Selecting compiler

On Windows you can ignore this.

On Linux and MacOS, must set `CXX` to a compiler. It should be Clang 18. (So usually `CXX=clang++` or `CXX=clang++-18`.)

There can be some ABI issues: if the MeshLib itself was built with GCC 13 or older or Clang 17 or older (note that Apple Clang uses a [different version numbering scheme](https://en.wikipedia.org/wiki/Xcode#Xcode_15.0_-_(since_visionOS_support)_2)), you must also append `-fclang-abi-compat=17` (always `17`) to the compiler string, e.g. `CXX='clang++ -fclang-abi-compat=17'`. Failure to do this will lead to undefined references to functions that have `requires ...` in their demangled names (demangle with `llvm-cxxfilt-18` if needed). This flag excludes `requires ...` from mangling, for compatibility with older compilers.

#### RAM usage vs build speed

There's a `NUM_FRAGMENTS=...` setting that lets you adjust compilation time vs RAM usage tradeoff.

Less fragments is faster, but requires more RAM.

The default value is `4`, which is good for 16 GB of RAM.

#### Selecting Python version

On Windows you don't need to care about this, we guess the version from the Vcpkg contents.

On Linux and Mac, you can pass e.g. `PYTHON_PKGCONF_NAME=python-3.10-embed` to force Python 3.10. The default is `python3-embed`, which uses some default Python version on Linux, and the latest one on Mac.

#### Selecting python package name

We currently default to `PACKAGE_NAME=meshlib2`, which is how the Python package will be called (`from meshlib2 import mrmeshpy`, etc).

You might want to set this to `PACKAGE_NAME=meshlib`.

#### Other useful flags

`--trace` flag will show all invoked shell commands. In general, all flags are forwarded to GNU Make that runs the makefile, which is what controls the generation.
