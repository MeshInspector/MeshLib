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

     * On the right panel, uncheck `Any Platform`, and enable only `Include Platforms`->`WebGL`.

     * Click `Apply`.

  3. Open `meshlib_vA.B.C.D_dotnet-wasm/native/emsdk-X.Y.Z/singlethreaded/` or `.../multithreaded/`.

     The `emsdk-X.Y.Z` version must match your Unity version. Consult [this table](https://docs.unity3d.com/6000.7/Documentation/Manual/webgl-native-plugins-with-emscripten.html) for which Unity version corresponds to which EMSDK version.

     The "singlethreaded"/"multithreaded" part must match your Wasm build settings in Unity. If you enabled Wasm multithreading in Unity, use the multithreaded version of the files. If you don't know what you're doing, you probably have a single-threaded build.

     * Select all files there (`.a` files).

     * On the right panel, similarly uncheck `Any Platform`, and enable only `Include Platforms`->`WebGL`.

     * Click `Apply`.

  4. Uncheck specific libraries that conflcit with Unity:

     * Select only `libblosc.a` and uncheck `WebGL`. Click `Apply`.

     Unity links some libraries of its own into the Wasm build, which conflict with ours, and need to be disabled.

     We still add them to the distribution for use outside of Unity.

  5. Uncheck the unused libraries:

     * If you used `.../singlethreaded/` libraries, then delete or disable the `.../multithreaded/` ones, or vice versa.

       Similarly, delete or disable all other `emsdk-X.Y.Z` versions, other than the one you're using.

     * To disable them (as an alternative to deleting the files), open the respective directory, select all files there (`.a` files), uncheck `Any Platform` and uncheck all platforms.
