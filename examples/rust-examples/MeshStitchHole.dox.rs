use meshlib::StitchHolesParams;

fn main() -> Result<(), String>
{
    // Load meshes
    let mut mesh = meshlib::mesh_load::from_any_supported_format("meshAwithHole.stl", None).into_result()?;
    let mesh_b = meshlib::mesh_load::from_any_supported_format("meshBwithHole.stl", None).into_result()?;

    // Unite meshes
    mesh.add_mesh(&mesh_b, None, None);

    // Find holes (expect that there are exactly 2 holes)
    let edges = mesh.topology().find_hole_representive_edges(None);
    if edges.size() != 2
    {
        return Err(format!("Expected 2 holes, found {}", edges.size()));
    }

    // Connect two holes
    let mut params = StitchHolesParams::new();
    params.set_metric(&meshlib::get_universal_metric(&mesh));
    meshlib::stitch_holes(&mut mesh, *edges.at(0), *edges.at(1), Some(&params));

    // Save result
    meshlib::mesh_save::to_any_supported_format(&mesh, "stitchedMesh.stl", None).into_result()?;

    Ok(())
}
