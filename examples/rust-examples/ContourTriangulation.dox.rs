fn main() -> Result<(), String>
{
    // load image as distance map
    let image = meshlib::image_load::from_any_supported_format("logo.jpg").into_result()?;
    let dm = meshlib::convert_image_to_distance_map(&image, None, None).into_result()?;

    let pl2 = meshlib::distance_map_to2_d_iso_polyline(&dm, 210.0);
    let triangulation_res = meshlib::planar_triangulation::triangulate_contours_std_vector_std_vector_mr_vector2f(&pl2.contours(None), None);

    meshlib::mesh_save::to_any_supported_format(&triangulation_res, "LogoMesh.ctm", None).into_result()?;

    Ok(())
}
