use meshlib::FillHoleParams;

fn main() -> Result<(), String>
{
    // Load mesh
    let mut mesh = meshlib::mesh_load::from_any_supported_format("mesh.stl", None).into_result()?;

    // Find single edge for each hole in mesh
    let hole_edges = mesh.topology().find_hole_representive_edges(None);

    // Setup filling parameters
    let mut params = FillHoleParams::new();
    params.set_metric(&meshlib::get_universal_metric(&mesh));

    // Alternatively, meshlib::fill_holes(&mut mesh, &hole_edges, Some(&params)) fills all holes at once.
    for i in 0..hole_edges.size()
    {
        meshlib::fill_hole(&mut mesh, *hole_edges.at(i), Some(&params));
    }

    // Save result
    meshlib::mesh_save::to_any_supported_format(&mesh, "filledMesh.stl", None).into_result()?;

    Ok(())
}
