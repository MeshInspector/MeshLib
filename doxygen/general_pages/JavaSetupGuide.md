# MeshLib Java Setup Guide {#MeshLibJavaSetupGuide}

# Setting Up MeshLib for Java Development

[TOC]

> [!NOTE]
> The Java bindings are new. They are generated from the C API and published as a source archive on the releases page, not on Maven Central yet.

The bindings are a plain jar, `meshlib.jar`, with no native code of its own: it calls the C API library `MeshLibC2` (see the [C Setup Guide](\ref MeshLibCSetupGuide)) through the foreign function API of Java 22, so there is no JNI glue to compile. C++ classes become classes whose instances own their C++ object (destroyed when garbage-collected, or earlier by `close()`), references returned by methods keep the object they point into alive, `std::string` parameters take `String`, and C++ exceptions become `meshlib.CppException`. The C++ namespaces become packages, so everything from `MR::` is under `meshlib.mr`, and the free functions are the static methods of `meshlib.mr.Functions`.

## Prerequisites {#JavaSetupPrerequisites}

 - A JDK 22 or newer (the bindings use `java.lang.foreign`, final since Java 22).
 - The MeshLib binaries for your platform, installed as described in the [C Setup Guide](\ref MeshLibCSetupGuide): the Linux `.deb` or vcpkg archive, or the macOS `.pkg`. On Windows use the DLLs of the NuGet package instead of the plain Windows archive, see below.

## Installation

 1. **Download the bindings**
 <br/> Open the [MeshLib GitHub Releases](https://github.com/MeshInspector/MeshLib/releases) page and download `meshlib_<version>_java_bindings.zip` of the same version as your MeshLib binaries. It holds `meshlib.jar` and the generated sources in `src`, which your IDE can use for navigation and documentation.

 2. **Put the jar on the class path**
 <br/> `javac -cp meshlib.jar ...` and `java -cp meshlib.jar:... ...` (`;` instead of `:` on Windows), or add it to your build tool as a plain jar dependency.

 3. **Make the MeshLib libraries loadable**
 <br/> Pass the directory with the MeshLib shared libraries in `java.library.path`, and enable native access for your code (otherwise the first call into the library prints a warning):
```sh
java --enable-native-access=ALL-UNNAMED -Djava.library.path=/usr/local/lib/MeshLib/lib -cp meshlib.jar:. MyProgram
```
  - Linux: `/usr/local/lib/MeshLib/lib` for the `.deb`, or `path_to_install/lib` for the vcpkg archive
  - macOS: `/Library/Frameworks/MeshLib.framework/Versions/Current/lib`
  - Windows: the `runtimes/win-x64/native` (or `win-arm64`) directory of the NuGet package `MeshLib.<version>.nupkg` from the releases page (a `.nupkg` is a zip). The plain Windows archive doesn't work from Java: `java.exe` has already loaded the JDK's own `zip.dll`, `freetype.dll` and `msvcp140.dll`, and MeshLib's DLLs import libraries of the same names, which the Windows loader then binds to the wrong modules. The NuGet package ships every dependency under a unique name, so there's no collision.

## Using the API

```java
import meshlib.mr.Functions;
import meshlib.mr.Mesh;
import meshlib.mr.expected_void_std_string;

public class Main
{
    public static void main(String[] args)
    {
        try (Mesh mesh = Functions.makeUVSphere(1.0f, 64, 64)) // Owns the C++ object, destroyed by `close()` or by the garbage collector.
        {
            System.out.println(mesh.topology().getValidFaces().count() + " faces");
            expected_void_std_string saved = meshlib.mr.meshsave.Functions.toAnySupportedFormat(mesh, "sphere.stl", null); // The C++ `expected<void, std::string>`.
            if (!saved.success())
                throw new RuntimeException("Failed to save: " + saved.error());
        }
    }
}
```

A few rules of thumb:

 - Parameters with default arguments, pointers and `std::optional` are nullable, pass `null` for the default. Scalars with defaults are boxed (`Integer`, `Float`), for the same reason.
 - Overloaded C++ functions keep their names as long as their Java parameter types differ; the others get the disambiguated names of the C API, e.g. `boolean_4_const_MR_Mesh_ref`. The doc comment of every method names the C function and the C++ original.
 - Callbacks (`std::function` parameters) are built from lambdas: `std_function_bool_from_float.of(progress -> true)`. An exception thrown by the lambda is rethrown by the method that called into MeshLib.
 - Downcasts and upcasts between a class and its bases are explicit: `objectMesh.upcastToObject()`, `ObjectMesh.dynamicDowncastFromObject(object)` (null if the object isn't an `ObjectMesh`).
 - Structs like `Vector3f` are views over native memory with the C++ layout: `v.x()`, `v.setX(1.0f)`, `v.copy()`.

## Try MeshLib with Java Examples

The [code samples](\ref Examples) have Java tabs for the [boolean](\ref ExampleMeshBoolean), [decimation](\ref ExampleMeshDecimate) and [load and save](\ref ExampleMeshLoadSave) examples, with the sources in `examples/java-examples` of the repository. Compile them with `javac --release 22 -cp meshlib.jar -d classes examples/java-examples/*.java` and run them with `java --enable-native-access=ALL-UNNAMED -Djava.library.path=<libraries> -cp meshlib.jar:classes MeshBoolean`.
