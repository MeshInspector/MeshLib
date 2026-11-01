use meshlib::{FaceBitSet, FaceId, Vector3f};

fn main() -> Result<(), String>
{
    // Load mesh
    let mut mesh = meshlib::mesh_load::from_any_supported_format("mesh.stl", None).into_result()?;

    // Select faces to extrude
    let mut faces_to_extrude = FaceBitSet::new();
    faces_to_extrude.auto_resize_set(FaceId {id_: 1}, None);
    faces_to_extrude.auto_resize_set(FaceId {id_: 2}, None);

    // Create duplicated verts on region boundary
    meshlib::make_degenerate_band_around_region(&mut mesh, &faces_to_extrude, None);

    // Find vertices that will be moved
    let verts_for_move = meshlib::get_incident_verts(mesh.topology(), &faces_to_extrude);

    let mut v = verts_for_move.find_first();
    while v.valid()
    {
        // Move each vertex
        meshlib::op_add_assign_mr_vector3f(mesh.points_mut().index_mut(v), &Vector3f::plus_z());
        v = verts_for_move.find_next(v);
    }

    // Invalidate internal caches after manual changing
    mesh.invalidate_caches(None);

    // Save mesh
    meshlib::mesh_save::to_any_supported_format(&mesh, "extrudedMesh.stl", None).into_result()?;

    Ok(())
}
