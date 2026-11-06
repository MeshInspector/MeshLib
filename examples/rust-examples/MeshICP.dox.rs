use meshlib::{AffineXf3f, ICPProperties, MeshOrPoints, ICP};

fn main() -> Result<(), String>
{
    // Load meshes
    let mut mesh_floating = meshlib::mesh_load::from_any_supported_format("meshA.stl", None).into_result()?;
    let mesh_fixed = meshlib::mesh_load::from_any_supported_format("meshB.stl", None).into_result()?;

    // Prepare ICP parameters
    let diagonal = mesh_fixed.get_bounding_box().diagonal();
    let icp_sampling_voxel_size = diagonal * 0.01; // To sample points from object
    let mut icp_params = ICPProperties::new();
    icp_params.set_dist_threshold_sq((diagonal * 0.1).powi(2)); // Use points pairs with maximum distance specified
    icp_params.set_exit_val(diagonal * 0.003); // Stop when distance reached

    // Calculate transformation
    let mut icp = ICP::construct_5(
        &MeshOrPoints::construct_mr_mesh(&mesh_floating),
        &MeshOrPoints::construct_mr_mesh(&mesh_fixed),
        &AffineXf3f::new(), &AffineXf3f::new(),
        icp_sampling_voxel_size);
    icp.set_params(&icp_params);
    let xf = icp.calculate_transformation();

    // Transform floating mesh
    mesh_floating.transform(&xf, None);

    // Output information string
    let info = icp.get_status_info();
    eprintln!("{}", *info);

    // Save result
    meshlib::mesh_save::to_any_supported_format(&mesh_floating, "meshA_icp.stl", None).into_result()?;

    Ok(())
}
