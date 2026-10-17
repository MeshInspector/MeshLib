
fn main() -> Result<(), String>
{
    // Load mesh.
    let mut mesh = meshlib::mesh_load::from_any_supported_format_2("mesh.stl", None).into_result().map_err(|error| format!("Failed to load mesh: {error}"))?;

    // Repack mesh optimally without building an AABB tree, which decimation invalidates anyway.
    // It's not necessary but highly recommended to achieve the best performance in parallel processing.
    mesh.pack_optimally_1(Some(false));

    // Setup decimate parameters.
    let mut settings = meshlib::DecimateSettings::new();

    // Decimation stop thresholds, you may specify one or both.
    settings.set_max_deleted_faces(1000); // Number of faces to be deleted.
    settings.set_max_error(0.05); // Maximum error when decimation stops.

    // Number of parts for parallel processing, greatly improves performance at the cost of minor quality loss.
    // 16 is a good choice even with fewer CPU cores, since smaller parts are processed faster; much larger values add per-part overhead.
    settings.set_subdivide_parts(16);

    // Decimate mesh.
    let result = meshlib::decimate_mesh(&mut mesh, Some(&settings));
    println!("Removed {} vertices, {} faces", result.verts_deleted(), result.faces_deleted());

    // Save result.
    meshlib::mesh_save::to_any_supported_format_3(&mesh, "decimated_mesh.stl", None).into_result().map_err(|error| format!("Failed to save mesh: {error}"))?;

    Ok(())
}
