use crate::assert_approx;
use meshlib::mr;
use meshlib::mr::{Vector3f, Vector3i};

fn assert_vec(v: &Vector3f, x: f32, y: f32, z: f32)
{
    assert_approx!(v.x, x, 0.0001);
    assert_approx!(v.y, y, 0.0001);
    assert_approx!(v.z, z, 0.0001);
}

#[test]
fn diagonal()
{
    assert_vec(&Vector3f::diagonal(3.0), 3.0, 3.0, 3.0);
}

#[test]
fn plus_x()
{
    assert_vec(&Vector3f::plus_x(), 1.0, 0.0, 0.0);
}

#[test]
fn plus_y()
{
    assert_vec(&Vector3f::plus_y(), 0.0, 1.0, 0.0);
}

#[test]
fn plus_z()
{
    assert_vec(&Vector3f::plus_z(), 0.0, 0.0, 1.0);
}

#[test]
fn add()
{
    let r = mr::op_add_mr_vector3f(&Vector3f {x: 1.0, y: 2.0, z: 3.0}, &Vector3f {x: 4.0, y: 5.0, z: 6.0});
    assert_vec(&r, 5.0, 7.0, 9.0);
}

#[test]
fn sub()
{
    let r = mr::op_sub_mr_vector3f(&Vector3f {x: 5.0, y: 7.0, z: 9.0}, &Vector3f {x: 4.0, y: 5.0, z: 6.0});
    assert_vec(&r, 1.0, 2.0, 3.0);
}

#[test]
fn mul_scalar()
{
    let r = mr::op_mul_mr_vector3f_float(&Vector3f {x: 1.0, y: 2.0, z: 3.0}, 2.0);
    assert_vec(&r, 2.0, 4.0, 6.0);
}

#[test]
fn length_sq()
{
    assert_approx!(Vector3f {x: 1.0, y: 2.0, z: 3.0}.length_sq(), 14.0, 0.0001);
}

#[test]
fn length()
{
    assert_approx!(Vector3f {x: 3.0, y: 4.0, z: 0.0}.length(), 5.0, 0.0001);
}

#[test]
fn diagonal_i()
{
    let v = Vector3i::diagonal(3);
    assert_eq!((v.x, v.y, v.z), (3, 3, 3));
}

#[test]
fn plus_x_i()
{
    let v = Vector3i::plus_x();
    assert_eq!((v.x, v.y, v.z), (1, 0, 0));
}

#[test]
fn plus_y_i()
{
    let v = Vector3i::plus_y();
    assert_eq!((v.x, v.y, v.z), (0, 1, 0));
}

#[test]
fn plus_z_i()
{
    let v = Vector3i::plus_z();
    assert_eq!((v.x, v.y, v.z), (0, 0, 1));
}
