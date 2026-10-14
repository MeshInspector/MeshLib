# MeshLib Rust Setup Guide {#MeshLibRustSetupGuide}

# Setting Up MeshLib for Rust Development

[TOC]

> [!NOTE]
> The Rust bindings are new. The crate is generated from the C API and is published as a source archive on the releases page, not on crates.io yet.

The crate is named `meshlib`. It wraps the C API library `MeshLibC2` (see the [C Setup Guide](\ref MeshLibCSetupGuide)) in safe Rust: C++ classes become types held in `Owned<T>`, which destroys the object when dropped, references and pointers become `&T` / `&mut T` / `Option<&T>`, `std::string` parameters take `&str`, and C++ exceptions become panics. The C++ namespaces become modules, so everything from `MR::` is under `meshlib::mr`.

## Prerequisites {#RustSetupPrerequisites}

 - [Rust](https://rustup.rs/) 1.82 or newer, with `cargo`.
 - The MeshLib binaries for your platform, installed as described in the [C Setup Guide](\ref MeshLibCSetupGuide): the Windows archive, the Linux `.deb` or vcpkg archive, or the macOS `.pkg`. The crate links against `MeshLibC2` (and `MeshLibC2Cuda`) from that installation.

## Installation

 1. **Download the crate**
 <br/> Open the [MeshLib GitHub Releases](https://github.com/MeshInspector/MeshLib/releases) page and download `meshlib_<version>_rust_bindings.zip` of the same version as your MeshLib binaries. Extract it next to your project, e.g. into `../MeshLibRust`. The archive holds `Cargo.toml`, `build.rs` and the generated `src`. The crate version is the MeshLib version with the fourth number as build metadata (`3.1.4+528` for v3.1.4.528).

 2. **Add the dependency**
 <br/> In your `Cargo.toml`:
```toml
[dependencies]
meshlib = { path = "../MeshLibRust" }
```

 3. **Point the linker to the MeshLib libraries**
 <br/> The crate's `build.rs` reads the `MESHLIB_LIB_DIR` environment variable and adds that directory to the linker search path. Set it to the directory with the MeshLib shared libraries when building:
  - Windows: `C:\meshlib-built\install\app\Release` (the `.dll` files; no import library is needed, the crate links with `raw-dylib`)
  - Linux: `/usr/local/lib/MeshLib/lib` for the `.deb`, or `path_to_install/lib` for the vcpkg archive
  - macOS: `/Library/Frameworks/MeshLib.framework/Versions/Current/lib`
```sh
MESHLIB_LIB_DIR=/usr/local/lib/MeshLib/lib cargo build
```

 4. **Make the libraries loadable at runtime**
 <br/> The same directory must be visible to the dynamic loader when your program runs: add it to `PATH` on Windows, `LD_LIBRARY_PATH` on Linux, `DYLD_LIBRARY_PATH` on macOS.

## Using the API

```rust
use meshlib::mr;

fn main()
{
    let mesh = mr::make_uv_sphere(Some(1.0), Some(64), Some(64)); // Owned<Mesh>, freed on drop.
    println!("{} faces", mesh.topology().get_valid_faces().count());
    let saved = mr::mesh_save::to_any_supported_format_3(&mesh, "sphere.stl", None); // Returns the C++ `expected<void, std::string>`.
    if let Some(error) = saved.error()
    {
        panic!("Failed to save: {error}");
    }
}
```

A few rules of thumb:

 - Parameters with default arguments take `Option`, pass `None` for the default.
 - Overloaded C++ functions keep the disambiguated names of the C API, e.g. `boolean_6`; hover the function in your editor, its doc comment names the C function and the C++ original.
 - Callbacks (`std::function` parameters) are built from closures: `mr::std_function_bool_from_float::from_fn(|progress| true)`.
 - Downcasts and upcasts between a class and its bases are explicit: `object_mesh.upcast_to_object()`, `mr::ObjectMesh::dynamic_downcast_from_object(&object)`.

## Try MeshLib with Rust Examples

The [code samples](\ref Examples) have Rust tabs for the [boolean](\ref ExampleMeshBoolean), [decimation](\ref ExampleMeshDecimate) and [load and save](\ref ExampleMeshLoadSave) examples, with the sources in `examples/rust-examples` of the repository. Build them with `cargo build --manifest-path examples/rust-examples/Cargo.toml` after setting `MESHLIB_LIB_DIR`.
