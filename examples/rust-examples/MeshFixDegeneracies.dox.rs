use meshlib::FixMeshDegeneraciesParams;

fn main() -> Result<(), String>
{
    let mut mesh = meshlib::mesh_load::from_any_supported_format("mesh.stl", None).into_result()?;

    // you can set various parameters for the resolving process; see the documentation for more info
    let mut params = FixMeshDegeneraciesParams::new();
    params.set_max_deviation(1e-5 * mesh.compute_bounding_box(None).diagonal());
    params.set_tiny_edge_length(1e-3);
    let _ = meshlib::fix_mesh_degeneracies(&mut mesh, &params);

    // Save result
    meshlib::mesh_save::to_any_supported_format(&mesh, "fixedMesh.stl", None).into_result()?;

    Ok(())
}
