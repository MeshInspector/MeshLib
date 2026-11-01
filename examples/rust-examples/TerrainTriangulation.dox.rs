use meshlib::{PointsLoadSettings, SaveSettings, VertColors};

fn main() -> Result<(), String>
{
    // load points
    let mut colors = VertColors::new();
    let mut pls = PointsLoadSettings::new();
    pls.set_colors(Some(&mut colors));
    let points = meshlib::points_load::from_any_supported_format("TerrainPoints.ply", Some(&pls)).into_result()?;
    let triangulation_res = meshlib::delaunay_triangulation_xy_const_mr_point_cloud_ref(&points, None).into_result()?;

    let mut ss = SaveSettings::new();
    if points.points().size() == colors.size()
    {
        ss.set_colors(Some(&colors));
    }
    meshlib::mesh_save::to_any_supported_format(&triangulation_res, "TerrainMesh.ctm", Some(&ss)).into_result()?;

    Ok(())
}
