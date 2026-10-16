use crate::common::make_torus;

#[test]
fn add_noise()
{
    let mut mesh = make_torus(1.1, 0.5, 16, 16);
    meshlib::add_noise_mr_mesh(&mut mesh, None, None);
}
