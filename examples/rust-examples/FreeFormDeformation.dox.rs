use meshlib::{FreeFormDeformer, Vector3i};

fn main() -> Result<(), String>
{
    // Load mesh
    let mut mesh = meshlib::mesh_load::from_any_supported_format("mesh.stl", None).into_result()?;

    // Construct deformer on mesh vertices
    let mut ff_deformer = FreeFormDeformer::construct_mr_mesh(&mut mesh, None);

    // Compute mesh bounding box
    let bbox = mesh.compute_bounding_box(None);

    // Init deformer with 3x3 grid on mesh box
    ff_deformer.init(Some(&Vector3i::diagonal(3)), Some(&bbox));

    // Move some control points of the grid to the center
    ff_deformer.set_ref_grid_point_position(&Vector3i {x: 1, y: 1, z: 0}, &bbox.center());
    ff_deformer.set_ref_grid_point_position(&Vector3i {x: 1, y: 1, z: 2}, &bbox.center());
    ff_deformer.set_ref_grid_point_position(&Vector3i {x: 0, y: 1, z: 1}, &bbox.center());
    ff_deformer.set_ref_grid_point_position(&Vector3i {x: 2, y: 1, z: 1}, &bbox.center());
    ff_deformer.set_ref_grid_point_position(&Vector3i {x: 1, y: 0, z: 1}, &bbox.center());
    ff_deformer.set_ref_grid_point_position(&Vector3i {x: 1, y: 2, z: 1}, &bbox.center());

    // Apply the deformation to the mesh vertices
    ff_deformer.apply();

    // Invalidate the mesh because of external vertex changes
    mesh.invalidate_caches(None);

    // Save deformed mesh
    meshlib::mesh_save::to_any_supported_format(&mesh, "deformed_mesh.stl", None).into_result()?;

    Ok(())
}
