// Points the linker to the MeshLib shared libraries: `MESHLIB_LIB_DIR` if set, else the ones bundled in `native/<os>-<arch>`
// (the release crate has them for Windows, Linux and macOS on x86_64 and aarch64). The crate itself links against nothing.
use std::path::{Path, PathBuf};

fn main()
{
    println!("cargo:rerun-if-env-changed=MESHLIB_LIB_DIR");
    if let Ok(dir) = std::env::var("MESHLIB_LIB_DIR")
    {
        if !dir.is_empty()
        {
            println!("cargo:rustc-link-search=native={dir}");
            return;
        }
    }

    let os = std::env::var("CARGO_CFG_TARGET_OS").unwrap();
    let arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap();
    let bundled = Path::new(&std::env::var("CARGO_MANIFEST_DIR").unwrap()).join("native").join(format!("{os}-{arch}"));
    println!("cargo:rerun-if-changed={}", bundled.display());
    if !bundled.is_dir()
    {
        return; // The sources-only crate: the libraries must be found by the linker some other way.
    }

    // `cargo run` and `cargo test` add to PATH / LD_LIBRARY_PATH / DYLD_FALLBACK_LIBRARY_PATH only the search paths inside the target directory,
    // hence the copy in OUT_DIR. Another copy goes next to the executables in `target/<profile>`, so that they also run without cargo.
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap()).join("lib");
    copy_libs(&bundled, &out_dir);
    println!("cargo:rustc-link-search=native={}", out_dir.display());
    if let Some(profile_dir) = out_dir.ancestors().nth(4) // target/<profile>/build/meshlib-<hash>/out/lib
    {
        copy_libs(&bundled, profile_dir);
    }
}

fn copy_libs(from: &Path, to: &Path)
{
    std::fs::create_dir_all(to).unwrap();
    for entry in std::fs::read_dir(from).unwrap()
    {
        let src = entry.unwrap().path();
        let dst = to.join(src.file_name().unwrap());
        if is_up_to_date(&src, &dst)
        {
            continue; // Also avoids replacing a library that a running program holds.
        }
        let _ = std::fs::remove_file(&dst);
        // A hard link saves copying ~100 MB per build directory, but needs the same volume.
        if std::fs::hard_link(&src, &dst).is_err()
        {
            std::fs::copy(&src, &dst).unwrap_or_else(|e| panic!("Failed to copy {} to {}: {e}", src.display(), dst.display()));
        }
    }
}

fn is_up_to_date(src: &Path, dst: &Path) -> bool
{
    match (std::fs::metadata(src), std::fs::metadata(dst))
    {
        (Ok(s), Ok(d)) => s.len() == d.len() && matches!((s.modified(), d.modified()), (Ok(s), Ok(d)) if d >= s),
        _ => false,
    }
}
