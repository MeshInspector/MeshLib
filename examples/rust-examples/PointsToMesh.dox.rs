fn main() -> Result<(), String>
{
    // load points
    let points = meshlib::points_load::from_any_supported_format("Points.ply", None).into_result()?;
    let triangulation_res = meshlib::triangulate_point_cloud(&points, None, None).into_option().expect("can be None only if canceled by progress callback");

    meshlib::mesh_save::to_any_supported_format(&triangulation_res, "Mesh.ctm", None).into_result()?;

    Ok(())
}
