# How to use MeshLib .NET bindings in Wasm?

MeshLib has a separate binary distribution for Wasm. The MeshLib NuGet package does not support Wasm. (In particular, because at the time of writing Nuget-for-Unity [doesn't support targeting Wasm](https://github.com/GlitchEnzo/NuGetForUnity/pull/756).)

Instead we provide a separate archive named `meshlib_vA.B.C.D_dotnet-wasm.zip`. Get it from the [Releases page](https://github.com/MeshInspector/MeshLib/releases).

## Instructions for Unity

### Disable the regular version of MeshLib in Wasm

If you also have installed MeshLib in Nuget-for-Unity (which is a good idea to be able to use MeshLib outside of Wasm, i.e. in the Unity editor and when targeting other platforms), then you must ensure it is not used for Wasm. To do that:

  1. In the `Project` tab (at the bottom of the screen by default), go to `Assets/Packages/MeshLib.A.B.C.D/lib/netstandard2.0/`

  2. Click on `MRDotNet2.dll` (extension not shown by default, it will have the "puzzle piece" icon).

     * On the right panel, enable `Exclude Platforms`->`WebGL`.

     * Click `Apply`.

### Add the Wasm version

  1. Unzip `meshlib_vA.B.C.D_dotnet-wasm.zip` and drag the contents into `Assets` in unity. They can be placed anywhere, but avoid putting them into `Packages/`, as Nuget-for-Unity can delete some files from there.

  2. Click on `meshlib_vA.B.C.D_dotnet-wasm/MRDotNet2Static.dll`.

     * On the right panel, uncheck `Any Platform`, and check `Include Platforms`->`WebGL`.

     * Click `Apply`.

  3. Open `meshlib_vA.B.C.D_dotnet-wasm/native/single/`.

  4. Select all files there (`.a` files).

     * On the right panel, similarly uncheck `Any Platform`, and check `Include Platforms`->`WebGL`.

     * Click `Apply`.

  5. Uncheck specific libraries that conflcit with Unity:

     * Select only `libblosc.a` and uncheck `WebGL`. Click `Apply`.

     Unity links some libraries of its own into the Wasm build, which conflict with ours, and need to be disabled.

     We still add them to the distribution for use outside of Unity.

## Different kinds of binaries

Right now we only ship one set of C++ binaries (`meshlib_vA.B.C.D_dotnet-wasm/native/single`). Those are single-threaded.

We plan to add more configurations (at least a multithreaded one), and possibly have versions built with different versions of Emscripten SDK (to match different versions of Unity).
