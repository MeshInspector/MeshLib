# Generating the bindings

First [install prerequisites](#1-install-prerequisites).

Then [compile MRBind](#2-compile-mrbind), our binding generator.

Then follow instructions for a specific language: [Python](#31-generate-python-bindings), [C](#32-generate-c-bindings), [C#](#33-generate-c-bindings).

## 1. Install prerequisites

Those are installed system-wide, this usually only needs to be done once, unless we update our dependencies.

### Windows

* Download https://vcpkg-export.s3.us-east-1.amazonaws.com/msys64_meshlib_mrbind.zip
* Unzip into `C:\msys64_meshlib_mrbind`

<details><summary>What does this do?</summary>

This installs MSYS2, which is a Linux-like build environment for Windows.

We use it to provide prebuilt Clang libraries for our parser, and also GNU Make to run our makefiles.

</details>

### Ubuntu

* Run `sudo scripts/mrbind/install_deps_ubuntu.sh`

<details><summary>What does this do?</summary>

This installs Clang. If the required version is missing in the stock repositories, it will add [the LLVM repository](https://apt.llvm.org/).

</details>

### MacOS

* Ensure you have [`brew`](https://brew.sh/) installed.

* Run `sudo scripts/mrbind/install_deps_macos.sh`

<details><summary>What does this do?</summary>

This uses Brew to install Clang and some other packages.

</details>

## 2. Compile MRBind

This needs to be done once per each copy of the MeshLib repository.

You need to repeat this step if we update MRBind.

### Windows

* Run `scripts/mrbind/install_mrbind_windows_msys2.bat`, either by double-clicking or in the terminal (**not** in the VS developer command prompt).

### Ubuntu

* Run `scripts/mrbind/install_mrbind_ubuntu.sh`

### MacOS

* Run `scripts/mrbind/install_mrbind_macos.sh`

## 3.1. Generate Python bindings

After building MeshLib, follow the OS-dependent steps below. After that we explain optional config flags.

The resulting Python modules are created next to the MeshLib shared libraries, in the `./meshlib` directory.

### Windows

* Open the `x64 Native Tools Command Prompt for VS` terminal in the Start menu.

* There, run <code>scripts\mrbind\generate_win.bat -B --trace VS_MODE=<b>\<see below\></b></code>

    * If you built MeshLib in Visual Studio in Release mode, pass `VS_MODE=Release`. If you built it in Debug mode, pass `VS_MODE=Debug`.
    * The current directory matters, the script will look for MeshLib in `./source/x64/Release` or `./source/x64/Debug` depending on `VS_MODE=...`.
    * If you built MeshLib in a different directory (e.g. if using CMake), also pass `MESHLIB_SHLIB_DIR=...`, pointing to the directory where `MRMesh.dll` was built.<br/>
    When using CMake, `VS_MODE` has to be set to `Debug` when `CMAKE_BUILD_TYPE` is `Debug`, and to `Release` otherwise. It controls which libraries (debug vs release) we use from Vcpkg.

### Linux

* Run <code>make -f scripts\mrbind\generate.mk -B --trace MESHLIB_SHLIB_DIR=<b>\<see below\></b></code>

    * `MESHLIB_SHLIB_DIR` must point to the directory containing `libMRMesh.so`, i.e. the CMake build directory plus `./bin`.

       It defaults to `build/Release/bin`. This assumes the CMake build directory is `build/Release`.

### MacOS

* Run `export PATH="$(brew --prefix)/opt/make/libexec/gnubin:$PATH"` to temporarily add a newer version of GNU Make to the PATH (now `make --version` should report 4.x or newer).

* Run <code>make -f scripts\mrbind\generate.mk -B --trace MESHLIB_SHLIB_DIR=<b>\<see below\></b></code>

    * `MESHLIB_SHLIB_DIR` must point to the directory containing `libMRMesh.dylib`, i.e. the CMake build directory plus `./bin`.

       It defaults to `build/Release/bin`. This assumes the CMake build directory is `build/Release`.

### Tunable flags for the generation script

* **`--trace` — enable verbose logs.** Remove to get quieter logs.

* **`-B` — force a full rebuild of the bindings.** Incremental builds are not very useful, because they're not perfect and can miss changes. Use incremental builds only e.g. when you're fixing linker errors.

* **`MODE=...` — optimization** setting for the Python module:

  * `release` (default) — Enable optimization, disable debug symbols. Same as what goes into the production.

  * `debug` — disable optimizations, enable debug symbols. Use this to debug the bindings.

  * `none` — neither enable optimizations nor enable the debug symbols. This gives the fastest build times, useful for testing bindings locally.

  You can also et entirely custom C++ compiler flags, by setting `EXTRA_CFLAGS` and `EXTRA_LDFLAGS`.

  The `EXTRA_...` flags are ignored for C#, and `MODE=none` is replaced with `MODE=debug` there.

* **`NUM_FRAGMENTS=?? -j??` — adjust RAM usage vs build speed tradeoff.**

  If you're running out of RAM, reduce `-j...`.

  `NUM_FRAGMENTS=??` is how many translation units the bindings are split into. `-j??` is the number of parallel build threads/processes. We have some heuristics to guess good values for this, they are printed when the script starts.

  Guessing the fastest combination isn't trivial. Usually you want to maximize threads (up to the number of cores), and then minimize the number of fragments as much as your RAM allows. The number of fragments should normally be a multiple of the number of threads.

* **`PYTHON_PKGCONF_NAME=python-3.??-embed` — select Python version.** We try to guess this one. You can set this to `python3-embed` to use whatever your OS considers to be the default version.

* **`ENABLE_CUDA=??` — enable or disable Cuda.** If you're building MeshLib without Cuda support, pass `ENABLE_CUDA=0` to skip the Cuda bindings too. It defaults to `1` everywhere except MacOS, where Cuda doesn't work.

  When this is disabled, a stub Cuda bindings are still generated, with a single function that reports that Cuda is not available.

### Troubleshooting Python bindings

* **`could not open 'MRMesh.lib': No such file or directory`**

  * MeshLib wasn't built, or `VS_MODE` is set incorrectly.

* **`machine type x86 conflicts with x64`**

  * You opened `x86 ...` VS developer command prompt, but we need `x64 Native`. Rebuild the bindings in x64 prompt.

* **`undefined symbol: void __cdecl std::_Literal_zero_is_expected(void)`**

  * Update your VS 2022.

* **Undefined references to MeshLib functions**.

  * This can only happen on Linux/Mac. If you look at the offending functions in the headers, they should have `requires` on them.

  * You likely used a non-default compiler when compiling MeshLib. Pass this compiler to `CXX_FOR_ABI=...` when generating Python bindings to fix the issue (e.g. `CXX_FOR_ABI=clang++-22` or `g++-15`).

    MeshLib is usually compiled using a different compiler than the Python bindings (for the bindings we use a specific version of Clang, the same that is used by the parser; using the same compiler for the bindings on all platforms makes writing them easier). Therefore, we must ensure that the two compilers use the same ABI. `CXX_FOR_ABI` should be set to the compiler the ABI of which we're trying to match. (Defaults to `CXX` environment variable, or `g++` if not set.) At the moment, if `CXX_FOR_ABI` is GCC 13 or older or Clang 17 or older (note that Apple Clang uses a [different version numbering scheme](https://en.wikipedia.org/wiki/Xcode#Xcode_15.0_-_(since_visionOS_support)_2)), we pass `-fclang-abi-compat=17` to our Clang 18 or newer (which is used to compile the bindings). This flag *disables* mangling of `requires` constraints into function names. If we guess incorrectly, you'll get undefined references to functions with `requires` constraints on them.

* **Importing the wheel segfaults on MacOS**

  * Make sure you're not linking against Python **and** make sure you use `-Xlinker -undefined -Xlinker dynamic_lookup` linker flags. The `generate.mk` should already do it correctly, but just keep this in mind. Also transitively linking Python seems to be fine (`-lMRPython` is fine).

    Failure to do this will have no effect when importing the module directly, but will segfault when importing it as a wheel, **or** when using a wrong Python version even without the wheel.

* **`cannot initialize type "expected_...": an object with that name is already defined`**

  Likely a conflict between `std::expected` and `tl::expected` (probably MRMesh ended up using the latter while MRBind is using the former). Try `EXTRA_CFLAGS='-DMB_PB11_ALLOW_STD_EXPECTED=0 -DMR_USE_STD_EXPECTED=0'` to make MRBind switch to `tl::expected`.

* **`lld-link: error: undefined symbol: void __cdecl std::_Literal_zero_is_expected(void)`**,
`>>> referenced by source/TempOutput/PythonBindings/x64/Release/binding.0.o:(public: __cdecl std::_Literal_zero::_Literal_zero<int>(int))`

  * This seems to be a VS2022 bug that's triggered by trying to bind `operator<=>` (taking its address?). We work around this by banning all `operator<=>`s with `--ignore`, see `mrbind_flags.txt`.

## 3.2. Generate C bindings

Running our script generates the code for the bindings, at `source/MeshLibC2` and `source/MeshLibC2Cuda`.

Then you must build MeshLib with a special CMake flag, which will build the generated bindings.

### Windows

* Open the `x64 Native Tools Command Prompt for VS` terminal in the Start menu.

* There, run <code>scripts\mrbind\generate_win.bat -B --trace TARGET=c</code>

* Compile MeshLib using CMake, with flag `-DMESHLIB_BUILD_GENERATED_C_BINDINGS=ON` to also compile the generated bindings.<br/>
  Compiling bindings using Visual Studio is not supported, you must use CMake.

### Linux

* Run <code>make -f scripts\mrbind\generate.mk -B --trace TARGET=c</code>

* Compile MeshLib using CMake, with flag `-DMESHLIB_BUILD_GENERATED_C_BINDINGS=ON` to also compile the generated bindings.<br/>

### MacOS

* Run `export PATH="$(brew --prefix)/opt/make/libexec/gnubin:$PATH"` to temporarily add a newer version of GNU Make to the PATH (now `make --version` should report 4.x or newer).

* Run <code>make -f scripts\mrbind\generate.mk -B --trace TARGET=c</code>

* Compile MeshLib using CMake, with flag `-DMESHLIB_BUILD_GENERATED_C_BINDINGS=ON` to also compile the generated bindings.<br/>

### Tunable flags for the generation script

* **`ENABLE_CUDA=??` — enable or disable Cuda.** If you're building MeshLib without Cuda support, pass `ENABLE_CUDA=0` to skip the Cuda bindings too. It defaults to `1` everywhere except MacOS, where Cuda doesn't work.

  When this is disabled, a stub Cuda bindings are still generated, with a single function that reports that Cuda is not available.

## 3.3. Generate C# bindings

You must [generate C bindings first](#32-generate-c-bindings), then continue from here.

To generate and build C# bindings, it's enough to generate C bindings without building them. But to actually run programs using the C# bindings, you must build the C bindings. You can do this in any order.

The steps below both generate the C# code (at `MeshLib/source/MRDotNet2`) and compile it.

### Windows

* Run <code>scripts\mrbind\generate_win.bat -B --trace TARGET=csharp</code>

* Locally running C# programs wasn't tested on this OS. You might need to copy the DLLs of the C bindings somewhere C# can find them.

### Linux

* Run <code>make -f scripts\mrbind\generate.mk -B --trace TARGET=csharp</code>

* To locally run C# programs that use our C# bindings, you might need to set environment variable `LD_LIBRARY_PATH=...` to the directory that contains `libMRMesh.so` when running your programs.

### MacOS

* Run `export PATH="$(brew --prefix)/opt/make/libexec/gnubin:$PATH"` to temporarily add a newer version of GNU Make to the PATH (now `make --version` should report 4.x or newer).

* Run <code>make -f scripts\mrbind\generate.mk -B --trace TARGET=csharp</code>

* Locally running C# programs wasn't tested on this OS. You might need to set environment variable `DYLD_LIBRARY_PATH=...` to the directory that contains `libMRMesh.dylib` when running your programs.

### Tunable generator flags

* **`MODE=??` — C# optimization mode.** Defaults to `release`, you can also pass `debug`.<br/>
  Most of the time this doesn't matter, since consuming C# bindings from another C# project will automatically rebuild them in the right mode (the one you use when building your project).

## 4. Other information

### Less common flags for the generator script

* **Selecting MRBind installation:** if you installed MRBind to a non-default location (the default is `./MeshLib/mrbind`), you must pass this location to `MRBIND_SOURCE=path/to/mrbind`.

    Additionally, if the MRBind binary is not at `$MRBIND_SOURCE/build/mrbind`, you must pass `MRBIND_EXE=...` (path to the executable itself, not its directory).

You can find more undocumented flags/variables in `generate.mk`.
