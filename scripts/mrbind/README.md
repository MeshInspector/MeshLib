## Generating the bindings

### On Windows:

Run those once:

* `scripts/mrbind/install_msys2_tools.bat` to install MSYS2 with Clang and libclang. Re-run to update to the latest version.

* `scripts/mrbind/install_mrbind.bat` to download and compile MRBind. Re-run to update to the latest version.

  Note that this downloads the latest `master`, while CI might be pointing to a specific commit. But it's simple to manually build a specific commit.

Lastly, open the VS developer command prompt (x64), and from it run `scripts/mrbind/generate_win.bat` to generate the bindings.

#### Current directory

The current directory must be the project root. Either the MeshLib root if this is a standalone MeshLib, or your project's root if it's a submodule in your project. The script will look for the MeshLib binaries in `./source/x64` relative to the current directory.

#### Debug vs release

By default we're using the `Debug` build of MeshLib. Use `.../generate_win.bat VS_MODE=Release` or `=Debug` to explicitly select the mode. Note, hovewer, that this doesn't affect the build settings of the bindings themselves.

#### Compiler settings

By default the bindings are built without optimizations and without debug symbols.

To enable optimizations: `.../generate_win.bat EXTRA_CFLAGS="-O3 -flto" EXTRA_LDFLAGS="-O3 -flto -s"`.

To enable debug symbols: `.../generate_win.bat EXTRA_CFLAGS=-g EXTRA_LDFLAGS=-g`.

#### RAM usage vs build speed

There's a `NUM_FRAGMENTS=...` setting that lets you adjust compilation time vs RAM usage tradeoff.

Less fragments is faster, but requires more RAM.

The default value is `4`, which is good for 16 GB of RAM.

#### Other useful flags

`--trace` flag will show all invoked shell commands. In general, all flags are forwarded to GNU Make that runs the makefile, which is what controls the generation.