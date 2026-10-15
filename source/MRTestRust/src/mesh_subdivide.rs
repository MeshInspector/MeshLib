use crate::common::make_torus;
use meshlib::mr;

#[test]
fn mesh_subdivide()
{
    let mut mesh = make_torus(1.1, 0.5, 16, 16);
    let mut settings = mr::SubdivideSettings::new();
    settings.set_max_deviation_after_flip(0.5);
    mr::subdivide_mesh_mr_mesh(&mut mesh, Some(&settings));
}
