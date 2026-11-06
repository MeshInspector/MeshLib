use crate::assert_approx;
use meshlib::Vector3f;

#[test]
fn make_cube()
{
    let cube = meshlib::make_cube(Some(&Vector3f {x: 1.0, y: 1.0, z: 1.0}), Some(&Vector3f {x: 0.0, y: 0.0, z: 0.0}));
    assert_approx!(cube.area_const_mr_face_bit_set_ptr(None), 6.0, 0.001);
}

#[test]
fn make_cylinder_advanced()
{
    let cylinder = meshlib::make_cylinder_advanced(Some(1.0), Some(0.5), Some(0.0), Some(2.0 * 3.1415), Some(1.0), Some(16));
    assert_approx!(cylinder.area_const_mr_face_bit_set_ptr(None), 10.541437, 0.1);
}

#[test]
fn make_torus()
{
    let torus = meshlib::make_torus(Some(2.0), Some(0.5), Some(32), Some(32), None);
    assert_approx!(torus.area_const_mr_face_bit_set_ptr(None), 39.257, 0.1);
}

#[test]
fn make_torus_with_self_intersections()
{
    let torus = meshlib::make_torus_with_self_intersections(Some(2.0), Some(0.5), Some(16), Some(16), None);
    assert_approx!(torus.area_const_mr_face_bit_set_ptr(None), 57.583, 0.01);
}

#[test]
fn make_sphere()
{
    let mut params = meshlib::SphereParams::new();
    params.set_radius(1.0);
    params.set_num_mesh_vertices(100);
    let sphere = meshlib::make_sphere(&params);
    assert_approx!(sphere.area_const_mr_face_bit_set_ptr(None), 12.157, 0.1);
}

#[test]
fn make_uv_sphere()
{
    let sphere = meshlib::make_uv_sphere(Some(1.5), Some(32), Some(32));
    assert_approx!(sphere.area_const_mr_face_bit_set_ptr(None), 28.15, 0.1);
}
