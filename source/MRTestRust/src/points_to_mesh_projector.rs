use crate::assert_approx;
use crate::common::create_cube;
use meshlib::mr;
use meshlib::mr::{AffineXf3f, Vector3f, VertId};

#[test]
fn find_signed_distances()
{
    let mesh_a = create_cube();
    let mesh_b = create_cube();
    let shift = Vector3f {x: 1.0, y: 2.0, z: 3.0};
    let xf = AffineXf3f::translation(&shift);
    let mut params = mr::MeshProjectionParameters::new();
    params.set_xf(Some(&xf));
    let results = mr::find_signed_distances_4(&mesh_a, &mesh_b, Some(&params), None);
    assert_eq!(results.size(), mesh_b.points().size());
    let mut max_dist = 0.0f32;
    for i in 0..results.size()
    {
        max_dist = max_dist.max(results.index(VertId {id_: i as i32}).abs());
    }
    assert_approx!(max_dist, shift.length(), 1e-6);
}
