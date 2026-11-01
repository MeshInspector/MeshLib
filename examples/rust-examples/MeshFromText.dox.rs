use meshlib::SymbolMeshParams;

fn main() -> Result<(), String>
{
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3
    {
        return Err("Usage: MeshFromText fontpath text".to_string());
    }
    let font_path = &args[1];
    let text = &args[2];

    // set text-to-mesh parameters
    let mut params = SymbolMeshParams::new();
    // input text
    params.set_text(text);
    // font file name
    params.set_path_to_font_file(font_path);
    let mesh = meshlib::create_symbols_mesh(&params).into_result().map_err(|error| format!("Failed to convert text to mesh: {error}"))?;

    meshlib::mesh_save::to_any_supported_format(&mesh, "mesh.ply", None).into_result().map_err(|error| format!("Failed to save result: {error}"))?;

    Ok(())
}
