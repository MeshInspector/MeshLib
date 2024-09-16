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

```
make -f scripts/mrbind/generate.mk -B
```

#### Current directory

The current directory must be the project root. Either the MeshLib root if this is a standalone MeshLib, or your project's root if it's a submodule in your project. The script will look for the MeshLib binaries in `./source/x64` relative to the current directory.

#### Choosing MeshLib build (Debug vs release)

On Windows:

* By default we're using the `Debug` build of MeshLib. Use `.../generate_win.bat VS_MODE=Release` or `=Debug` to explicitly select the mode. Note, hovewer, that this doesn't affect the build settings of the bindings themselves.

On Linux and MacOS:

* Must pass `MESHLIB_SHLIB_DIR=...` (as Make flag, not env variable) pointing to your build `bin` directory, e.g. `build/RelWithDebugInfo/bin`.

#### Full vs incremental build

`-B` forces a full rebuild. Removing this flag in theory performs an incremental build, but changing anything in MeshLib headers requires a full rebuild anyway. Removing this flag is only useful for rebuilding auxiliary parts of the module (copying `__init__.py` again, and so on).

#### Compiler settings

By default the bindings are built without optimizations and without debug symbols.

To enable optimizations: `.../generate_win.bat EXTRA_CFLAGS="-Oz -flto=thin" EXTRA_LDFLAGS="-Oz -flto=thin -s"`.

To enable debug symbols: `.../generate_win.bat EXTRA_CFLAGS=-g EXTRA_LDFLAGS=-g`.

#### RAM usage vs build speed

There's a `NUM_FRAGMENTS=...` setting that lets you adjust compilation time vs RAM usage tradeoff.

Less fragments is faster, but requires more RAM.

The default value is `4`, which is good for 16 GB of RAM.

#### Other useful flags

`--trace` flag will show all invoked shell commands. In general, all flags are forwarded to GNU Make that runs the makefile, which is what controls the generation.
