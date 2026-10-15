# MeshLib Rust Setup Guide {#MeshLibRustSetupGuide}

# Setting Up MeshLib for Rust Development

[TOC]

> [!NOTE]
> The Rust bindings are new. The crate is generated from the C API and is published as archives on the releases page, not on crates.io yet.

The crate is named `meshlib`. It wraps the C API library `MeshLibC2` (see the [C Setup Guide](\ref MeshLibCSetupGuide)) in safe Rust: C++ classes become types held in `Owned<T>`, which destroys the object when dropped, references and pointers become `&T` / `&mut T` / `Option<&T>`, `std::string` parameters take `&str`, and C++ exceptions become panics. The C++ namespaces become modules, so everything from `MR::` is under `meshlib::mr`.

## Prerequisites {#RustSetupPrerequisites}

 - [Rust](https://rustup.rs/) 1.88 or newer, with `cargo`.
 - Windows, Linux (glibc 2.28 or newer) or macOS, on x86_64 or aarch64.

## Installation

 1. **Download the crate**
 <br/> Open the [MeshLib GitHub Releases](https://github.com/MeshInspector/MeshLib/releases) page and download one of:
  - `meshlib_<version>_rust_bindings.zip` (4 MB): `Cargo.toml`, `build.rs` and the generated `src`. On the first build, `build.rs` downloads the MeshLib shared libraries of the target platform (25-35 MB) from the same release, checks their SHA256 and keeps them in the user's cache directory (`%LOCALAPPDATA%\meshlib`, `~/Library/Caches/meshlib`, `~/.cache/meshlib`; set `MESHLIB_CACHE_DIR` to change it). Offline, set `MESHLIB_NATIVE_URL` to a directory with the downloaded `meshlib_<version>_rust_native_<os>-<arch>.zip`.
  - `meshlib_<version>_rust.zip` (160 MB): the same plus, in `native`, the libraries of all supported platforms, so nothing is downloaded.

 Extract it next to your project, e.g. into `../MeshLibRust`. The libraries are the same ones as in the NuGet package. The crate version is the MeshLib version with the fourth number as build metadata (`3.1.4+528` for v3.1.4.528).

 2. **Add the dependency**
 <br/> In your `Cargo.toml`:
```toml
[dependencies]
meshlib = { path = "../MeshLibRust" }
```

That is all: `cargo build`, `cargo run` and `cargo test` find the libraries of the target platform by themselves. The crate's `build.rs` also puts them next to your executables in `target/<profile>`, which is enough to run them without `cargo` on Windows. On Linux, let the executable find them there by adding to your own `build.rs`:
```rust
println!("cargo:rustc-link-arg-bins=-Wl,-rpath,$ORIGIN");
```
On macOS, set `DYLD_FALLBACK_LIBRARY_PATH` to that directory. When you ship your program, ship the libraries with it.

### Using your own MeshLib installation {#RustSetupOwnLibraries}

Instead of the bundled or downloaded libraries, the crate can use the MeshLib binaries you already have, installed as described in the [C Setup Guide](\ref MeshLibCSetupGuide): the Windows archive, the Linux `.deb` or vcpkg archive, or the macOS `.pkg` of the same version. Point the linker to them with the `MESHLIB_LIB_DIR` environment variable, then nothing is downloaded:
  - Windows: `C:\meshlib-built\install\app\Release` (the `.dll` files; no import library is needed, the crate links with `raw-dylib`)
  - Linux: `/usr/local/lib/MeshLib/lib` for the `.deb`, or `path_to_install/lib` for the vcpkg archive
  - macOS: `/Library/Frameworks/MeshLib.framework/Versions/Current/lib`
```sh
MESHLIB_LIB_DIR=/usr/local/lib/MeshLib/lib cargo build
```
The same directory must be visible to the dynamic loader when your program runs: add it to `PATH` on Windows, `LD_LIBRARY_PATH` on Linux, `DYLD_LIBRARY_PATH` on macOS.

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
