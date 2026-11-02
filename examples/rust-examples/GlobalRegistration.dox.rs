use meshlib::{AffineXf3f, Box3f, ICPProperties, MeshOrPoints, MeshOrPointsXf, MultiwayICP, MultiwayICPSamplingParameters, ObjId, PointCloud};

fn print_stats(icp: &MultiwayICP)
{
    println!("Samples: {}", icp.get_num_samples());
    println!("Active point pairs: {}", icp.get_num_active_pairs());
    if icp.get_num_active_pairs() > 0
    {
        let p2pt_metric = icp.get_mean_sq_dist_to_point(None);
        let p2pl_metric = icp.get_mean_sq_dist_to_plane(None);
        println!("RMS point-to-point distance: {} ± {}", p2pt_metric, icp.get_mean_sq_dist_to_point(Some(p2pt_metric as f64)));
        println!("RMS point-to-plane distance: {} ± {}", p2pl_metric, icp.get_mean_sq_dist_to_plane(Some(p2pl_metric as f64)));
    }
}

fn main() -> Result<(), String>
{
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4
    {
        return Err(format!("Usage: {} INPUT1 INPUT2 [INPUTS...] OUTPUT", args[0]));
    }

    // the global registration can be applied to meshes and point clouds
    // to simplify the sample app, we will work with point clouds only
    let mut inputs = Vec::new();
    // as ICP and MultiwayICP classes accept both meshes and point clouds,
    // the input data must be converted to special wrapper objects
    // NB: the wrapper objects hold *references* to the source data, NOT their copies
    let mut objects = meshlib::Vector_MR_MeshOrPointsXf_MR_ObjId::new();
    let mut max_bbox = Box3f::new();
    for input_path in &args[1..args.len() - 1]
    {
        let point_cloud = meshlib::points_load::from_any_supported_format(input_path, None).into_result().map_err(|error| format!("Failed to load point cloud: {error}"))?;

        let bbox = point_cloud.compute_bounding_box(None);
        if !max_bbox.valid() || bbox.volume() > max_bbox.volume()
        {
            max_bbox = bbox;
        }

        // you may also set an affine transformation for each input as a second argument
        objects.push_back_const_mr_mesh_or_points_xf_ref(&MeshOrPointsXf::construct_from(&MeshOrPoints::construct_mr_point_cloud(&point_cloud), AffineXf3f::new()));
        inputs.push(point_cloud);
    }

    // you can set various parameters for the global registration; see the documentation for more info
    let mut sampling_params = MultiwayICPSamplingParameters::new();
    // set sampling voxel size
    sampling_params.set_sampling_voxel_size(max_bbox.diagonal() * 0.03);
    let mut icp = MultiwayICP::construct(&objects, &sampling_params);

    icp.set_params(&ICPProperties::new());

    // gather statistics
    icp.update_all_point_pairs(None);
    print_stats(&icp);

    println!("Calculating transformations...");
    let xfs = icp.calculate_transformations(None);
    print_stats(&icp);

    let mut output = PointCloud::new();
    for (i, input) in inputs.iter().enumerate()
    {
        let xf = xfs.index(ObjId {id_: i as i32});
        let points = input.points();
        for j in 0..points.size()
        {
            output.add_point(&xf.call(points.index(meshlib::VertId {id_: j as i32})));
        }
    }

    meshlib::points_save::to_any_supported_format(&output, &args[args.len() - 1], None).into_result().map_err(|error| format!("Failed to save point cloud: {error}"))?;

    Ok(())
}
