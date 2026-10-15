// Points the linker to the C bindings library, if `MESHLIB_LIB_DIR` is set. The consumers of the crate need it, the crate itself links against nothing.
fn main()
{
    println!("cargo:rerun-if-env-changed=MESHLIB_LIB_DIR");
    if let Ok(dir) = std::env::var("MESHLIB_LIB_DIR")
    {
        if dir.is_empty() {return;}
        println!("cargo:rustc-link-search=native={dir}");
    }
}
