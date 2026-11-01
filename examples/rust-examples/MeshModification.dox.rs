use meshlib::{AffineXf3f, Matrix3f, MeshRelaxParams, SubdivideSettings, Vector3f};

fn main()
{
    let mut mesh = meshlib::make_torus(None, None, None, None, None);

    // Relax mesh (5 iterations)
    let mut relax_params = MeshRelaxParams::new();
    relax_params.set_iterations(5);
    meshlib::relax(&mut mesh, Some(&relax_params), None);

    // Subdivide mesh
    let mut props = SubdivideSettings::new();
    props.set_max_deviation_after_flip(0.5);
    meshlib::subdivide_mesh_mr_mesh(&mut mesh, Some(&props));

    // Rotate mesh
    let rotation_xf = AffineXf3f::linear(&Matrix3f::rotation_float(&Vector3f::plus_z(), std::f32::consts::PI * 0.5));
    mesh.transform(&rotation_xf, None);
}
