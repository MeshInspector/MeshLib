use meshlib::mr;

fn main() -> Result<(), String>
{
    // Load mesh.
    let mut loaded = mr::mesh_load::from_any_supported_format_2("mesh.stl", None);
    let Some(mesh) = loaded.value_mut() else
    {
        return Err(format!("Failed to load mesh: {}", loaded.error().unwrap()));
    };

    // Setup decimate parameters.
    let mut settings = mr::DecimateSettings::new();

    // Decimation stop thresholds, you may specify one or both.
    settings.set_max_deleted_faces(1000); // Number of faces to be deleted.
    settings.set_max_error(0.05); // Maximum error when decimation stops.

    // Number of parts for parallel processing, greatly improves performance at the cost of minor quality loss.
    // A value close to the number of CPU cores is usually the fastest; much larger values add per-part overhead.
    settings.set_subdivide_parts(16);

    // Decimate mesh.
    let result = mr::decimate_mesh(mesh, Some(&settings));
    println!("Removed {} vertices, {} faces", result.verts_deleted(), result.faces_deleted());

    // Save result.
    let saved = mr::mesh_save::to_any_supported_format_3(mesh, "decimated_mesh.stl", None);
    if let Some(error) = saved.error()
    {
        return Err(format!("Failed to save mesh: {error}"));
    }

    Ok(())
}
