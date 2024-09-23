## Generating the bindings

### Windows prerequisites:

Run those once:

* `scripts/mrbind/install_msys2_tools.bat` to install MSYS2 with Clang and libclang. Re-run to update to the latest version.

* `scripts/mrbind/install_mrbind.bat` to download and compile MRBind. Re-run to update to the latest version.

  Note that this downloads the latest `master`, while CI might be pointing to a specific commit. But it's simple to manually build a specific commit.

### MacOS prerequisites:

* **Install some helper tools:**
  ```sh
  brew install make gawk
  export PATH="/opt/homebrew/opt/make/libexec/gnubin:$PATH"
  ```
  If already installed, only run the second line to add it to PATH.

* **Install LLVM/Clang:**
  ```sh
  brew install llvm
  export PATH="/opt/homebrew/opt/llvm/bin:$PATH"
  ```
  If already installed, only run the second line to add it to PATH.

* **Build MRBind:**
  ```sh
  git clone https://github.com/MeshInspector/mrbind ~/mrbind
  cd ~/mrbind
  CC=clang CXX=clang++ cmake -B build -DCMAKE_PREFIX_PATH=/opt/homebrew/opt/llvm
  cmake --build build
  ```


### Creating the bindings:

You need to run:
* On Windows: `scripts/mrbind/generate_win.bat` from VS developer command prompt (x64).
* On Linux and MacOS: `make -f scripts/mrbind/generate.mk`

You must pass some additional flags.

Here are some example configurations, but read the docs below for details.

* Windows: `scripts/mrbind/generate_win.bat -B --trace VS_MODE=Release` (or `=Debug`)

* MacOS: `make -f scripts/mrbind/generate.mk -B --trace CXX='clang++ -fclang-abi-compat=17' MRBIND_SOURCE=/Users/user/repos/mrbind MESHLIB_SHLIB_DIR=build/RelWithDebugInfo/bin PYTHON_PKGCONF_NAME=python-3.10-embed`

* Ubuntu: `make -f scripts/mrbind/generate.mk -B --trace CXX='clang++-18 -fclang-abi-compat=17' MRBIND_SOURCE=/home/user/repos/mrbind MESHLIB_SHLIB_DIR=build/RelWithDebugInfo/bin` (drop `-fclang-abi-compat=17` on Ubuntu 24.04 and newer).

#### Selecting MeshLib build

On Windows:

* The current directory matters. We look for MeshLib in `./source/x64/$VS_MODE` relative to the current directory.

  By default `VS_MODE=Debug`. You can pass `VS_MODE=Release` as a flag to Make (note, not as an env variable).

On Linux and MacOS:

* Must pass `MESHLIB_SHLIB_DIR=...` (as a Make flag, not env variable) pointing to your build `bin` directory, e.g. `build/RelWithDebugInfo/bin`.

#### Full vs incremental build

`-B` forces a full rebuild. Removing this flag in theory performs an incremental build, but changing anything in MeshLib headers requires a full rebuild anyway. Removing this flag is only useful for rebuilding auxiliary parts of the module (copying `__init__.py` again, and so on).

#### Selecting MRBind installation

On Windows, if you used `install_mrbind.bat`, you don't need to worry about this.

On Other platforms, you must pass `MRBIND_SOURCE=path/to/mrbind` to Make (note, not as an env variable), pointing to the MRBind source directory.

If MRBind is not at `$MRBIND_SOURCE/build/mrbind`, must also pass `MRBIND_EXE` pointing to the executable.

#### Compiler settings

By default the bindings are built without optimizations and without debug symbols.

To enable optimizations: `.../generate_win.bat EXTRA_CFLAGS="-Oz -flto=thin" EXTRA_LDFLAGS="-Oz -flto=thin -s"`. (Omit `-s` on MacOS, it seems to have no effect, and the linker warns you shouldn't use it.)

To enable debug symbols: `.../generate_win.bat EXTRA_CFLAGS=-g EXTRA_LDFLAGS=-g`.

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

#### Other useful flags

`--trace` flag will show all invoked shell commands. In general, all flags are forwarded to GNU Make that runs the makefile, which is what controls the generation.
