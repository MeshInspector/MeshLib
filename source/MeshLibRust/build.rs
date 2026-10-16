// Points the linker to the MeshLib shared libraries, the first found of:
//   - `MESHLIB_LIB_DIR`, a MeshLib installation;
//   - `native/<os>-<arch>`, bundled in `meshlib_<version>_rust.zip`;
//   - the libraries of the target platform downloaded once per user from the release listed in `native.txt` (the crate without libraries).
// The crate itself links against nothing.
use std::io::Read;
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
    if std::env::var_os("DOCS_RS").is_some()
    {
        return; // No network there, and nothing to link.
    }

    let platform = format!("{}-{}", std::env::var("CARGO_CFG_TARGET_OS").unwrap(), std::env::var("CARGO_CFG_TARGET_ARCH").unwrap());
    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let bundled = manifest_dir.join("native").join(&platform);
    let native_txt = manifest_dir.join("native.txt");
    let libs = if bundled.is_dir()
    {
        println!("cargo:rerun-if-changed={}", bundled.display());
        bundled
    }
    else if native_txt.is_file()
    {
        println!("cargo:rerun-if-changed={}", native_txt.display());
        download(&native_txt, &platform)
    }
    else
    {
        return; // A local build of the repository: the libraries must be found by the linker some other way.
    };

    // `cargo run` and `cargo test` add to PATH / LD_LIBRARY_PATH / DYLD_FALLBACK_LIBRARY_PATH only the search paths inside the target directory,
    // hence the copy in OUT_DIR. Another copy goes next to the executables in `target/<profile>`, so that they also run without cargo.
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap()).join("lib");
    copy_libs(&libs, &out_dir);
    println!("cargo:rustc-link-search=native={}", out_dir.display());
    if let Some(profile_dir) = out_dir.ancestors().nth(4) // target/<profile>/build/meshlib-<hash>/out/lib
    {
        copy_libs(&libs, profile_dir);
    }
}

// `native.txt` is the line `url <release download URL>`, then `<sha256> <archive name>` per platform.
// `MESHLIB_NATIVE_URL` replaces the URL: another http(s) location or a local directory with the archives.
// The libraries are unpacked into `MESHLIB_CACHE_DIR` (default: the user's cache directory)`/meshlib/<archive name without .zip>`.
fn download(native_txt: &Path, platform: &str) -> PathBuf
{
    println!("cargo:rerun-if-env-changed=MESHLIB_NATIVE_URL");
    println!("cargo:rerun-if-env-changed=MESHLIB_CACHE_DIR");
    let text = std::fs::read_to_string(native_txt).unwrap();
    let mut url = text.lines().find_map(|l| l.strip_prefix("url ")).expect("No URL in native.txt").trim().to_string();
    let suffix = format!("_{platform}.zip");
    let (sha256, name) = text.lines().filter_map(|l| l.split_once(' ')).find(|(_, name)| name.trim().ends_with(&suffix))
        .unwrap_or_else(|| panic!("MeshLib has no libraries for {platform}"));
    let (sha256, name) = (sha256.trim(), name.trim());

    let dir = cache_dir().join("meshlib").join(name.strip_suffix(".zip").unwrap());
    if dir.join(".complete").is_file()
    {
        return dir;
    }

    if let Ok(u) = std::env::var("MESHLIB_NATIVE_URL")
    {
        if !u.is_empty() {url = u;}
    }
    let archive = if url.starts_with("http://") || url.starts_with("https://")
    {
        let full_url = format!("{}/{name}", url.trim_end_matches('/'));
        println!("cargo:warning=Downloading the MeshLib libraries from {full_url}");
        let mut data = Vec::new();
        ureq::get(&full_url).call().and_then(|r| Ok(r.into_body().into_reader().read_to_end(&mut data)?))
            .unwrap_or_else(|e| panic!("Failed to download {full_url}: {e}\nOffline, set MESHLIB_LIB_DIR to the MeshLib libraries, or MESHLIB_NATIVE_URL to a directory with {name}."));
        data
    }
    else
    {
        let path = Path::new(&url).join(name);
        std::fs::read(&path).unwrap_or_else(|e| panic!("Failed to read {}: {e}", path.display()))
    };

    use sha2::Digest;
    let actual: String = sha2::Sha256::digest(&archive).iter().map(|b| format!("{b:02x}")).collect();
    if actual != sha256
    {
        panic!("{name}: SHA256 is {actual}, expected {sha256}");
    }

    // Unpack next to the final directory and rename, so that a concurrent or interrupted build never sees a partial one.
    std::fs::create_dir_all(dir.parent().unwrap()).unwrap();
    let tmp = dir.with_extension(format!("tmp{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    zip::ZipArchive::new(std::io::Cursor::new(archive)).and_then(|mut z| z.extract(&tmp))
        .unwrap_or_else(|e| panic!("Failed to unpack {name}: {e}"));
    std::fs::write(tmp.join(".complete"), "").unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    if std::fs::rename(&tmp, &dir).is_err() && !dir.join(".complete").is_file()
    {
        panic!("Failed to move {} to {}", tmp.display(), dir.display());
    }
    let _ = std::fs::remove_dir_all(&tmp); // When another build was first.
    dir
}

fn cache_dir() -> PathBuf
{
    let var = |name| std::env::var_os(name).filter(|v| !v.is_empty()).map(PathBuf::from);
    if let Some(dir) = var("MESHLIB_CACHE_DIR")
    {
        return dir;
    }
    let dir = if cfg!(windows) {var("LOCALAPPDATA")}
        else if cfg!(target_os = "macos") {var("HOME").map(|h| h.join("Library/Caches"))}
        else {var("XDG_CACHE_HOME").or_else(|| var("HOME").map(|h| h.join(".cache")))};
    dir.expect("No cache directory for the MeshLib libraries, set MESHLIB_CACHE_DIR")
}

fn copy_libs(from: &Path, to: &Path)
{
    std::fs::create_dir_all(to).unwrap();
    for entry in std::fs::read_dir(from).unwrap()
    {
        let src = entry.unwrap().path();
        if !src.is_file() || src.file_name().unwrap() == ".complete"
        {
            continue;
        }
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
