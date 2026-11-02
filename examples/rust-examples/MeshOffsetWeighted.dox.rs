use meshlib::{MeshPart, VertId, VertScalars};

fn main() -> Result<(), String>
{
    // Create some mesh
    let mesh = meshlib::make_torus(None, None, None, None, None);

    // Create VertScalars obj with weights for every vertex
    let vert_size = mesh.topology().vert_size();
    let mut scalars = VertScalars::construct(vert_size);
    for i in 0..vert_size
    {
        let v = VertId {id_: i as i32};
        *scalars.index_mut(v) = (mesh.points().index(v).x / 5.0).abs(); // Individual extra offset sizes for points
    }

    let mut params = meshlib::weighted_shell::ParametersMetric::new();
    // Algorithm is voxel based, voxel size affects performance and form of result mesh
    params.set_voxel_size(meshlib::suggest_voxel_size(&MeshPart::construct(&mesh, None), 1000.0));
    // common basic offset applied for all point
    // Vertex-specific weighted offsets applied after the basic one
    params.set_offset(0.2);
    let max_weight = (0..vert_size).map(|i| *scalars.index(VertId {id_: i as i32})).fold(f32::MIN, f32::max);
    params.dist_mut().set_max_weight(max_weight); // should always have maximum between weights provided

    let result = meshlib::weighted_shell::mesh_shell(&mesh, &scalars, &params).into_result()?;

    meshlib::mesh_save::to_any_supported_format(&result, "offset_weighted.ctm", None).into_result()?;

    Ok(())
}
