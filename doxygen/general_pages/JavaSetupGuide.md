# MeshLib Java Setup Guide {#MeshLibJavaSetupGuide}

# Setting Up MeshLib for Java Development

[TOC]

The bindings are a plain jar, `meshlib.jar`, with no native code of its own: it calls the C API library `MeshLibC2` (see the [C Setup Guide](\ref MeshLibCSetupGuide)) through the foreign function API of Java 22, so there is no JNI glue to compile. C++ classes become classes whose instances own their C++ object (destroyed when garbage-collected, or earlier by `close()`), references returned by methods keep the object they point into alive, `std::string` parameters take `String`, and C++ exceptions become `meshlib.CppException`. A function returning `Expected<T>` returns `T` (nothing for `Expected<void>`) and throws `meshlib.UnexpectedResultException` with the error message, like in Python and C#. Everything from `MR::` is directly in the `meshlib` package, nested namespaces become subpackages (`MR::MeshLoad` is `meshlib.meshload`), and the free functions are the static methods of the `Functions` class of each package.

## Prerequisites {#JavaSetupPrerequisites}

 - A JDK 22 or newer (the bindings use `java.lang.foreign`, final since Java 22).

## Installation from Maven Central

The bindings are the artifact `com.meshinspector:meshlib`, and the MeshLib libraries of each platform are artifacts of the same name and version with the classifiers `natives-windows-x86_64`, `natives-windows-aarch64`, `natives-linux-x86_64`, `natives-linux-aarch64`, `natives-macos-x86_64` and `natives-macos-aarch64`. On the first use the bindings extract the libraries of the current platform from the natives jar on the class path to the temporary directory (or to the directory in the system property `meshlib.native.cache`), and later runs reuse them; nothing else needs to be installed.

**Gradle** (`build.gradle.kts`), with the natives of the platforms you need:
```kotlin
dependencies {
    implementation("com.meshinspector:meshlib:1.2.3.4")
    runtimeOnly("com.meshinspector:meshlib:1.2.3.4:natives-windows-x86_64")
    runtimeOnly("com.meshinspector:meshlib:1.2.3.4:natives-linux-x86_64")
}
```

**Maven**: the dependency on `meshlib`, and one on the natives with a classifier picked by an OS profile, as in [examples/java-examples/pom.xml](https://github.com/MeshInspector/MeshLib/blob/master/examples/java-examples/pom.xml):
```xml
<dependency>
  <groupId>com.meshinspector</groupId>
  <artifactId>meshlib</artifactId>
  <version>1.2.3.4</version>
</dependency>
<dependency>
  <groupId>com.meshinspector</groupId>
  <artifactId>meshlib</artifactId>
  <version>1.2.3.4</version>
  <classifier>natives-${meshlib.platform}</classifier>
  <scope>runtime</scope>
</dependency>
```

Run with `--enable-native-access=ALL-UNNAMED` (otherwise the first call into the library prints a warning).

## Installation from the release archive

 1. **Download the bindings**
 <br/> Open the [MeshLib GitHub Releases](https://github.com/MeshInspector/MeshLib/releases) page and download `meshlib_<version>_java_bindings.zip` of the same version as your MeshLib binaries. It holds `meshlib.jar` and the generated sources in `src`, which your IDE can use for navigation and documentation.

 2. **Put the jar on the class path**
 <br/> `javac -cp meshlib.jar ...` and `java -cp meshlib.jar:... ...` (`;` instead of `:` on Windows), or add it to your build tool as a plain jar dependency.

 3. **Make the MeshLib libraries loadable**
 <br/> Install the MeshLib binaries for your platform as described in the [C Setup Guide](\ref MeshLibCSetupGuide) (the Linux `.deb` or vcpkg archive, or the macOS `.pkg`; on Windows use the DLLs of the NuGet package instead of the plain Windows archive, see below). Pass the directory with the MeshLib shared libraries in `java.library.path`, and enable native access for your code:
```sh
java --enable-native-access=ALL-UNNAMED -Djava.library.path=/usr/local/lib/MeshLib/lib -cp meshlib.jar:. MyProgram
```
  - Linux: `/usr/local/lib/MeshLib/lib` for the `.deb`, or `path_to_install/lib` for the vcpkg archive
  - macOS: `/Library/Frameworks/MeshLib.framework/Versions/Current/lib`
  - Windows: the `runtimes/win-x64/native` (or `win-arm64`) directory of the NuGet package `MeshLib.<version>.nupkg` from the releases page (a `.nupkg` is a zip). The plain Windows archive doesn't work from Java: `java.exe` has already loaded the JDK's own `zip.dll`, `freetype.dll` and `msvcp140.dll`, and MeshLib's DLLs import libraries of the same names, which the Windows loader then binds to the wrong modules. The NuGet package ships every dependency under a unique name, so there's no collision.

## Using the API

```java
import meshlib.Functions;
import meshlib.Mesh;

public class Main
{
    public static void main(String[] args)
    {
        try (Mesh mesh = Functions.makeUVSphere(1.0f, 64, 64)) // Owns the C++ object, destroyed by `close()` or by the garbage collector.
        {
            System.out.println(mesh.topology().getValidFaces().count() + " faces");
            meshlib.meshsave.Functions.toAnySupportedFormat(mesh, "sphere.stl", null); // Returns `Expected<void>` in C++: an error is thrown as `UnexpectedResultException`.
        }
    }
}
```

A few rules of thumb:

 - Parameters with default arguments, pointers and `std::optional` are nullable, pass `null` for the default. Scalars with defaults are boxed (`Integer`, `Float`), for the same reason.
 - Overloaded C++ functions keep their names as long as their Java parameter types differ; the others get the disambiguated names of the C API, e.g. `boolean_move`. The doc comment of every method names the C function and the C++ original.
 - Callbacks (`std::function` parameters) are built from lambdas: `std_function_bool_from_float.of(progress -> true)`. An exception thrown by the lambda is rethrown by the method that called into MeshLib.
 - Downcasts and upcasts between a class and its bases are explicit: `objectMesh.upcastToObject()`, `ObjectMesh.dynamicDowncastFromObject(object)` (null if the object isn't an `ObjectMesh`).
 - Structs like `Vector3f` are views over native memory with the C++ layout: `v.x()`, `v.setX(1.0f)`, `v.copy()`.

## Try MeshLib with Java Examples

The [code samples](\ref Examples) have Java tabs for the [boolean](\ref ExampleMeshBoolean), [decimation](\ref ExampleMeshDecimate) and [load and save](\ref ExampleMeshLoadSave) examples, with the sources in `examples/java-examples` of the repository. Run them with Maven from that directory: `mvn compile exec:exec -Dmeshlib.version=1.2.3.4 -Dexample=MeshBoolean`. Or compile them with `javac --release 22 -cp meshlib.jar -d classes examples/java-examples/*.java` and run them with `java --enable-native-access=ALL-UNNAMED -Djava.library.path=<libraries> -cp meshlib.jar:classes MeshBoolean`.
