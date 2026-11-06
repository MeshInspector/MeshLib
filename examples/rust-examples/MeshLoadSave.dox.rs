
fn main() -> Result<(), String>
{
    let args: Vec<String> = std::env::args().collect();
    let input_path = args.get(1).map_or("mesh.stl", String::as_str);
    let output_path = args.get(2).map_or("mesh.ply", String::as_str);

    // Load mesh. `into_result()` turns the returned `expected` into a Rust `Result`.
    let mesh = meshlib::mesh_load::from_any_supported_format(input_path, None).into_result().map_err(|error|
    {
        if args.len() <= 1
        {
            eprintln!("Usage: MeshLoadSave [input mesh] [output mesh]");
        }
        format!("Failed to load mesh: {error}")
    })?;

    // Save mesh.
    meshlib::mesh_save::to_any_supported_format(&mesh, output_path, None).into_result().map_err(|error| format!("Failed to save mesh: {error}"))?;

    Ok(())
}
