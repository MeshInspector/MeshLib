# Generating the bindings

[Installing prerequisites](#installing-prerequisites) — [Generating bindings](#generating-bindings) — [Troubleshooting](#troubleshooting-python-bindings)

## Installing prerequisites

Run **`scripts/mrbind/install_deps_<platform>`** to install the dependencies (on Linux and MacOS - as root), then **`scripts/mrbind/install_mrbind_<platform>`** to build MRBind (not at root). MRBind is our binding generator.

You can re-run those scripts to update the dependencies and/or MRBind itself.

Among other things, the scripts do following:

* On Ubuntu, they may add [the LLVM repository](https://apt.llvm.org/) to install the specific version of Clang and libclang that we need.

* On Windows, install MSYS2 to `C:\msys64_meshlib_mrbind`.

More details on what the scripts do on different platforms:

<details><summary><b>Windows</b></summary>

* **Installing dependencies:**

    On Windows we use MSYS2, because it provides prebuilt libclang and provides GNU Make to run our makefile.

    MSYS2 is a package manager, roughly speaking. They provide a bunch of MinGW-related packages (compilers and prebuilt libraries). Luckily Clang can always cross-compile, so MSYS2's MinGW Clang can produce MSVC-compatible executables with the correct flags. You still need to have VS installed though, since it will use its libraries.

    We use [MSYS2 CLANG64](https://www.msys2.org/docs/environments/) environment. Consult `install_deps_windows_msys2.bat` for the list of packages we install in it.

    We don't use the latest Clang version, instead we download and install the version specified in `clang_version_msys2.txt`.

* **Building MRBind:**

    MRBind source code is at https://github.com/MeshInspector/mrbind/.

    We build MRBind at `MeshLib/mrbind`, but you can build it [elsewhere](#less-common-flags) manually.

    We build in [MSYS2 CLANG64](https://www.msys2.org/docs/environments/) environment, using MSYS2's Clang. Other compilers are not guaranteed to work.


</details>

<details><summary><b>Ubuntu</b></summary>

* **Installing dependencies:**

    We want a certain version of Clang (see `clang_version.txt`), and since older versions of Ubuntu don't have it, we add Clang's repository: https://apt.llvm.org

    And obviously we install some packages, see `install_deps_ubuntu.sh` for the list.

* **Building MRBind:**

    MRBind source code is at https://github.com/MeshInspector/mrbind/.

    We build MRBind at `MeshLib/mrbind`, but you can build it [elsewhere](#less-common-flags) manually.

    You might want to pass `-DClang_DIR=/usr/lib/cmake/clang-VERSION` (where `VERSION` is the one mentioned in `clang_version.txt`) if you have several versions of libclang installed, because otherwise CMake might pick an arbitrary one (apparently it picks the first one returned by globbing `clang-*`, which might not be the latest one).

    Use `CC=clang-VERSION CXX=clang++-VERSION cmake ....` to build using Clang. Other compilers might work, but that's not guaranteed.

</details>

<details><summary><b>MacOS</b></summary>

* **Installing dependencies:**

    Homebrew must already be installed.

    We install a certain version of Clang and libclang from it (see `clang_version.txt`), and also GNU Make and Gawk. MacOS has its own Make, but it's outdated. It seems to have Gawk, but we install our own just in case.

    What we install from Brew is the regular Clang, not Apple Clang (Apple's fork for Clang), because that is based on an outdated branch of Clang.

    You must run following to add the installed things to your PATH. On Arm Macs:
    ```sh
    export PATH="/opt/homebrew/opt/make/libexec/gnubin:$PATH"
    export PATH="/opt/homebrew/opt/llvm/bin@VERSION:$PATH" # See the correct VERSION in `clang_version.txt`.
    ```
    And on x86 Macs the installation directory seems to be `/usr/local/...` instead of `/opt/homebrew/...`.

* **Building MRBind:**

    MRBind source code is at https://github.com/MeshInspector/mrbind/.

    We build MRBind at `MeshLib/mrbind`, but you can build it [elsewhere](#less-common-flags) manually.

    Make sure your PATH is correct, as explained in the previous step.

    Use `CC=clang-VERSION CXX=clang++-VERSION cmake ....` to build using the Clang we've installed. It might build using Apple Clang as well (if you don't set your PATH as explained above), but that's not guaranteed to work. (If you want to try it, you must pass `-DCMAKE_PREFIX_PATH=/opt/homebrew/opt/llvm` for it to find our libclang.)

</details>

## Generating bindings

**Python**: If you're generating Python bindings, you should build MeshLib first, then run the generator, which will take care of building the bindings.

**C**: If you're generating C bindings, run the generator first, and *then* build MeshLib along with the freshly generated bindings. Building C bindings is only supported with CMake, not MSBuild. Pass `-DMESHLIB_BUILD_GENERATED_C_BINDINGS=ON` to CMake to build the bindings (in addition to the rest of MeshLib). The generated bindings are human-readable and are located in `source/MeshLibC2`.

How to run the generator on different platforms:

* **On Windows:** `scripts\mrbind\generate_win.bat -B --trace` from the VS developer command prompt (use the `x64 Native` one!).

  When generating the Python bindings, the current directory matters, as this will look for MeshLib in `./source/x64/Release`. Add `VS_MODE=Debug` at the end if you built MeshLib in debug mode.

  The `generate_win.bat` file merely calls `generate.mk` (see below) inside of MSYS2 shell. You can use that directly if you want.

* **On Linux:** `make -f scripts/mrbind/generate.mk -B --trace`

  This will look for MeshLib in `./build/Release/bin`. Pass `MESHLIB_SHLIB_DIR=path/to/bin` for a different directory.

* **On MacOS:** Same as on Linux, but before running the command you must adjust the PATH. On Arm Macs: `export PATH="/opt/homebrew/opt/make/libexec/gnubin:$PATH"`, and on x86 Macs `/usr/local/...` instead of `/opt/homebrew/...`. This adds the version of Make installed in Homebrew to PATH, because the default one is outdated. Confirm the version with `make --version`, it must be 4.x or newer.

### Some common flags:

* **`--trace` — enable verbose logs.**

* **`-B` — force a full rebuild of the bindings.** Incremental builds are not very useful, because they're not perfect and can miss changes. Use incremental builds e.g. when you're fixing linker errors.

The remaining flags are for Python bindings only:

* **`MODE=none` — disable optimization** for faster build times. The default is `MODE=release`. To enable debug symbols, use `MODE=debug`. To set completely custom compiler flags, set `EXTRA_CFLAGS` and `EXTRA_LDFLAGS`.

* **`NUM_FRAGMENTS=?? -j??` — adjust RAM usage vs build speed tradeoff.** `NUM_FRAGMENTS=??` is how many translation units the bindings are split into. `-j??` is the number of parallel build threads/processes. `NUM_FRAGMENTS=64 -j8` is the default, good for 16 GB of RAM.

  Guessing the fastest combination isn't trivial. Usually less fragments and more threads lead to faster builds but more RAM usage, but not always; turns out `NUM_FRAGMENTS=1` isn't optimal even if you have enough RAM for it.

* **`PYTHON_PKGCONF_NAME=python-3.??-embed` — select Python version.** We try to guess this one. You can set this to `python3-embed` to use whatever the OS considers to be the default version.

### Selecting the compiler:

For simplicity, we compile the Python bindings with the same Clang that we use for parsing the code. (Consult `clang_version.txt` for the current version.) But you can override this using `CXX_FOR_BINDINGS`.

`CXX_FOR_BINDINGS` has an additional use that matters for both Python and C. We use it to locate the "Clang resource directory", which the parser needs. The variable must be set to the same version of Clang that provides libclang that was used to build MRBind, otherwise you might get compatibility issues. But normally we should be able to guess the value of this variable, so normally you don't have to think abou tthis.

**ABI compatibility (Python only):** Since MeshLib is compiled using a different compiler, we must ensure the two use the same ABI. `CXX_FOR_ABI` should be set to the compiler the ABI of which we're trying to match. (Defaults to `CXX` environment variable, or `g++` if not set.) At the moment, if `CXX_FOR_ABI` is GCC 13 or older or Clang 17 or older (note that Apple Clang uses a [different version numbering scheme](https://en.wikipedia.org/wiki/Xcode#Xcode_15.0_-_(since_visionOS_support)_2)), we pass `-fclang-abi-compat=17` to our Clang 18 or newer. This flag *disables* mangling `requires` constraints into function names. If we guess incorrectly, you'll get undefined references to functions with `requires` constraints on them.

### Less common flags:

* **Selecting MRBind installation:** if you installed MRBind to a non-default location (`MeshLib/mrbind`), you must pass this location to `MRBIND_SOURCE=path/to/mrbind`.

    Additionally, if the MRBind binary is not at `$MRBIND_SOURCE/build/mrbind`, you must pass `MRBIND_EXE=...` (path to the executable itself, not its directory).

You can find some undocumented flags/variables in `generate.mk`.

## Troubleshooting Python bindings

* **`could not open 'MRMesh.lib': No such file or directory`**

  * MeshLib wasn't built, or `VS_MODE` is set incorrectly.

* **`machine type x86 conflicts with x64`**

  * You opened `x86 ...` VS developer command prompt, but we need `x64 Native`. Rebuild the bindings in x64 prompt.

* **`undefined symbol: void __cdecl std::_Literal_zero_is_expected(void)`**

  * Update your VS 2022.

* **Importing the wheel segfaults on MacOS**

  * Make sure you're not linking against Python **and** do use `-Xlinker -undefined -Xlinker dynamic_lookup` linker flags. The `generate.mk` should already do it correctly, just keep this in mind. Also transitively linking Python seems to be fine (`-lMRPython` is fine).

    Failure to do this will have no effect when importing the module directly, but will segfault when importing it as a wheel, **or** when using a wrong Python version even without the wheel.

* **`cannot initialize type "expected_...": an object with that name is already defined`**

  Likely a conflict between `std::expected` and `tl::expected` (probably MRMesh ended up using the latter while MRBind is using the former). Try `EXTRA_CFLAGS='-DMB_PB11_ALLOW_STD_EXPECTED=0 -DMR_USE_STD_EXPECTED=0'` to make MRBind switch to `tl::expected`.

* **`lld-link: error: undefined symbol: void __cdecl std::_Literal_zero_is_expected(void)`**,
`>>> referenced by source/TempOutput/PythonBindings/x64/Release/binding.0.o:(public: __cdecl std::_Literal_zero::_Literal_zero<int>(int))`

  * This seems to be a VS2022 bug that's triggered by trying to bind `operator<=>` (taking its address?). We work around this by banning all `operator<=>`s with `--ignore`, see `mrbind_flags.txt`.
