use meshlib::mr;
use meshlib::mr::{AffineXf3f, Vector3f};

#[test]
fn mesh_mesh_distance()
{
    let sphere1 = mr::make_uv_sphere(Some(1.0), Some(8), Some(8));
    let whole_sphere1 = mr::MeshPart::construct(&sphere1, None);
    let d11 = mr::find_distance(&whole_sphere1, &whole_sphere1, None, Some(f32::MAX));
    assert_eq!(*d11.dist_sq(), 0.0);

    let z_shift = AffineXf3f::translation(&Vector3f {x: 0.0, y: 0.0, z: 3.0});
    let d1z = mr::find_distance(&whole_sphere1, &whole_sphere1, Some(&z_shift), Some(f32::MAX));
    assert_eq!(*d1z.dist_sq(), 1.0);

    let sphere2 = mr::make_uv_sphere(Some(2.0), Some(8), Some(8));
    let whole_sphere2 = mr::MeshPart::construct(&sphere2, None);
    let d12 = mr::find_distance(&whole_sphere1, &whole_sphere2, None, Some(f32::MAX));
    let dist12 = d12.dist_sq().sqrt();
    assert!(dist12 > 0.9 && dist12 < 1.0);
}
