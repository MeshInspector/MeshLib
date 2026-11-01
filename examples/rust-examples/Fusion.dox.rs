fn main() -> Result<(), String>
{
    // load points
    let points = meshlib::points_load::from_any_supported_format("Points.ply", None).into_result()?;

    let mut params = meshlib::PointsToMeshParameters::new();
    let voxel_size = points.compute_bounding_box(None).diagonal() * 1e-2;
    params.set_voxel_size(voxel_size);
    params.set_sigma(voxel_size.max(meshlib::find_avg_points_radius(&points, 50, None)));
    params.set_min_weight(1.0);

    let mesh = meshlib::points_to_mesh_fusion(&points, &params).into_result()?;

    meshlib::mesh_save::to_any_supported_format(&mesh, "Mesh.ctm", None).into_result()?;

    Ok(())
}
