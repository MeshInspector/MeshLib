use crate::assert_approx;
use meshlib::mr;
use meshlib::mr::{Matrix3f, Vector3f};

fn matrix(rows: [[f32; 3]; 3]) -> Matrix3f
{
    let v = |r: [f32; 3]| Vector3f {x: r[0], y: r[1], z: r[2]};
    Matrix3f {x: v(rows[0]), y: v(rows[1]), z: v(rows[2])}
}

fn assert_matrix(m: &Matrix3f, rows: [[f32; 3]; 3])
{
    let actual = [[m.x.x, m.x.y, m.x.z], [m.y.x, m.y.y, m.y.z], [m.z.x, m.z.y, m.z.z]];
    for i in 0..3
    {
        for j in 0..3
        {
            assert_approx!(actual[i][j], rows[i][j], 0.001);
        }
    }
}

#[test]
fn identity()
{
    assert_matrix(&Matrix3f::identity(), [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
}

#[test]
fn rotation_scalar()
{
    let m = Matrix3f::rotation_float(&Vector3f {x: 1.0, y: 0.0, z: 0.0}, 3.14159 / 2.0); // 90 degrees around the X axis
    assert_approx!(m.x.x, 1.0, 0.001);
    assert_approx!(m.y.y, 0.0, 0.001);
    assert_approx!(m.y.z, -1.0, 0.001);
    assert_approx!(m.z.y, 1.0, 0.001);
    assert_approx!(m.z.z, 0.0, 0.001);
}

#[test]
fn rotation_vector()
{
    let m = Matrix3f::rotation_mr_vector3f(&Vector3f {x: 1.0, y: 0.0, z: 0.0}, &Vector3f {x: 0.0, y: 1.0, z: 0.0});
    assert_matrix(&m, [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]);
}

#[test]
fn add()
{
    let a = matrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);
    let b = matrix([[9.0, 8.0, 7.0], [6.0, 5.0, 4.0], [3.0, 2.0, 1.0]]);
    assert_matrix(&mr::op_add_mr_matrix3f(&a, &b), [[10.0; 3]; 3]);
}

#[test]
fn sub()
{
    let a = matrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);
    let b = matrix([[9.0, 8.0, 7.0], [6.0, 5.0, 4.0], [3.0, 2.0, 1.0]]);
    assert_matrix(&mr::op_sub_mr_matrix3f(&a, &b), [[-8.0, -6.0, -4.0], [-2.0, 0.0, 2.0], [4.0, 6.0, 8.0]]);
}

#[test]
fn mul()
{
    let a = Matrix3f::identity();
    let b = matrix([[2.0, 1.0, 0.0], [1.0, 2.0, 0.0], [0.0, 0.0, 3.0]]);
    assert_matrix(&mr::op_mul_mr_matrix3f(&a, &b), [[2.0, 1.0, 0.0], [1.0, 2.0, 0.0], [0.0, 0.0, 3.0]]);
}

#[test]
fn mul_vector()
{
    let m = matrix([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]]);
    let r = mr::op_mul_mr_matrix3f_mr_vector3f(&m, &Vector3f {x: 1.0, y: 2.0, z: 3.0});
    assert_approx!(r.x, 2.0, 0.001);
    assert_approx!(r.y, 4.0, 0.001);
    assert_approx!(r.z, 6.0, 0.001);
}

#[test]
fn equal()
{
    let a = Matrix3f::identity();
    let b = Matrix3f::identity();
    let c = matrix([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]]);
    assert!(mr::op_equal_mr_matrix3f(&a, &b));
    assert!(!mr::op_equal_mr_matrix3f(&a, &c));
}
