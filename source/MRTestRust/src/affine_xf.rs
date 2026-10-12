use crate::assert_approx;
use meshlib::mr;
use meshlib::mr::{AffineXf3f, Matrix3f, Vector3f};

fn matrix(rows: [[f32; 3]; 3]) -> Matrix3f
{
    let v = |r: [f32; 3]| Vector3f {x: r[0], y: r[1], z: r[2]};
    Matrix3f {x: v(rows[0]), y: v(rows[1]), z: v(rows[2])}
}

#[test]
fn new()
{
    let result = AffineXf3f::new();
    assert!(mr::op_equal_mr_matrix3f(&result.A, &Matrix3f::identity()));
    assert_approx!(result.b.x, 0.0, 0.0001);
    assert_approx!(result.b.y, 0.0, 0.0001);
    assert_approx!(result.b.z, 0.0, 0.0001);
}

#[test]
fn translation()
{
    let result = AffineXf3f::translation(&Vector3f {x: 1.0, y: 2.0, z: 3.0});
    assert!(mr::op_equal_mr_matrix3f(&result.A, &Matrix3f::identity()));
    assert_approx!(result.b.x, 1.0, 0.0001);
    assert_approx!(result.b.y, 2.0, 0.0001);
    assert_approx!(result.b.z, 3.0, 0.0001);
}

#[test]
fn linear()
{
    let linear = matrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);
    let result = AffineXf3f::linear(&linear);
    assert!(mr::op_equal_mr_matrix3f(&result.A, &linear));
    assert_approx!(result.b.x, 0.0, 0.0001);
    assert_approx!(result.b.y, 0.0, 0.0001);
    assert_approx!(result.b.z, 0.0, 0.0001);
}

#[test]
fn mul()
{
    let xf1 = AffineXf3f::translation(&Vector3f {x: 1.0, y: 2.0, z: 3.0});
    let xf2 = AffineXf3f::linear(&matrix([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]]));
    let result = mr::op_mul_mr_affine_xf3f(&xf1, &xf2);
    let expected_a = mr::op_mul_mr_matrix3f(&xf1.A, &xf2.A);
    let expected_b = Vector3f {
        x: xf1.A.x.x * xf2.b.x + xf1.A.y.x * xf2.b.y + xf1.A.z.x * xf2.b.z + xf1.b.x,
        y: xf1.A.x.y * xf2.b.x + xf1.A.y.y * xf2.b.y + xf1.A.z.y * xf2.b.z + xf1.b.y,
        z: xf1.A.x.z * xf2.b.x + xf1.A.y.z * xf2.b.y + xf1.A.z.z * xf2.b.z + xf1.b.z,
    };
    assert!(mr::op_equal_mr_matrix3f(&result.A, &expected_a));
    assert_approx!(result.b.x, expected_b.x, 0.0001);
    assert_approx!(result.b.y, expected_b.y, 0.0001);
    assert_approx!(result.b.z, expected_b.z, 0.0001);
}

#[test]
fn apply()
{
    let xf = AffineXf3f::translation(&Vector3f {x: 1.0, y: 2.0, z: 3.0});
    let point = Vector3f {x: 4.0, y: 5.0, z: 6.0};
    let result = xf.call(&point);
    let expected = Vector3f {
        x: xf.A.x.x * point.x + xf.A.y.x * point.y + xf.A.z.x * point.z + xf.b.x,
        y: xf.A.x.y * point.x + xf.A.y.y * point.y + xf.A.z.y * point.z + xf.b.y,
        z: xf.A.x.z * point.x + xf.A.y.z * point.y + xf.A.z.z * point.z + xf.b.z,
    };
    assert_approx!(result.x, expected.x, 0.0001);
    assert_approx!(result.y, expected.y, 0.0001);
    assert_approx!(result.z, expected.z, 0.0001);
}
