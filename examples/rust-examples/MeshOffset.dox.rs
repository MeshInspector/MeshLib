use meshlib::{GeneralOffsetParameters, MeshPart, SignDetectionMode};

fn main() -> Result<(), String>
{
    // Create mesh
    let mesh = meshlib::make_cube(None, None);
    let mesh_part = MeshPart::construct(&mesh, None);

    // Setup parameters
    let mut params = GeneralOffsetParameters::new();
    // calculate voxel size depending on desired accuracy and/or memory consumption
    params.set_voxel_size(meshlib::suggest_voxel_size(&mesh_part, 10000000.0));
    if !meshlib::find_right_boundary(mesh.topology(), None, None).empty()
    {
        params.set_sign_detection_mode(SignDetectionMode::HoleWindingRule); // use if you have holes in mesh
    }

    // Make offset mesh
    let offset = mesh.compute_bounding_box(None).diagonal() * 0.1;
    let result = meshlib::general_offset_mesh(&mesh_part, offset, &params).into_result()?;

    // Save result
    meshlib::mesh_save::to_any_supported_format(&result, "mesh_offset.stl", None).into_result()?;

    Ok(())
}
