use meshlib::{MeshPart, OffsetParameters, PointCloud, UniformSamplingSettings, Vector3f};
use std::f32::consts::PI;

fn main() -> Result<(), String>
{
    // Generate point cloud
    let mut pc = PointCloud::new();
    pc.points_mut().reserve(10000);
    for i in 0..100
    {
        let u = 2.0 * PI * i as f32 / (100.0 - 1.0);
        for j in 0..100
        {
            let v = PI * j as f32 / (100.0 - 1.0);

            pc.add_point(&Vector3f {
                x: u.cos() * v.sin(),
                y: u.sin() * v.sin(),
                z: v.cos(),
            });
        }
    }
    // Remove duplicate points
    let mut sampling_settings = UniformSamplingSettings::new();
    sampling_settings.set_distance(1e-3);
    let vs = meshlib::point_uniform_sampling(&pc, &sampling_settings).into_option().expect("can be None only if canceled by progress callback");
    pc.set_valid_points(&vs);
    pc.invalidate_caches();

    // Triangulate it
    let triangulated = meshlib::triangulate_point_cloud(&pc, None, None).into_option().expect("can be None only if canceled by progress callback");

    // Fix possible issues
    let triangulated_part = MeshPart::construct(&triangulated, None);
    let mut offset_params = OffsetParameters::new();
    offset_params.set_voxel_size(meshlib::suggest_voxel_size(&triangulated_part, 5e+6));
    let mesh = meshlib::offset_mesh(&triangulated_part, 0.0, Some(&offset_params)).into_result()?;

    // Save result
    meshlib::mesh_save::to_any_supported_format(&mesh, "result.stl", None).into_result()?;

    Ok(())
}
