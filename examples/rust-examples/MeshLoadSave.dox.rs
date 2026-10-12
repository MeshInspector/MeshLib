use meshlib::mr;

fn main() -> Result<(), String>
{
    let args: Vec<String> = std::env::args().collect();
    let input_path = args.get(1).map_or("mesh.stl", String::as_str);
    let output_path = args.get(2).map_or("mesh.ply", String::as_str);

    // Load mesh.
    let loaded = mr::mesh_load::from_any_supported_format_2(input_path, None);
    let Some(mesh) = loaded.value() else
    {
        if args.len() <= 1
        {
            eprintln!("Usage: MeshLoadSave [input mesh] [output mesh]");
        }
        return Err(format!("Failed to load mesh: {}", loaded.error().unwrap()));
    };

    // Save mesh.
    let saved = mr::mesh_save::to_any_supported_format_3(mesh, output_path, None);
    if let Some(error) = saved.error()
    {
        return Err(format!("Failed to save mesh: {error}"));
    }

    Ok(())
}
