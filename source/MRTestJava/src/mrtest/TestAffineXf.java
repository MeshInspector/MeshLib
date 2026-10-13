package mrtest;

import static mrtest.Common.check;
import static mrtest.Common.checkApprox;

import meshlib.mr.AffineXf3f;
import meshlib.mr.Functions;
import meshlib.mr.Matrix3f;
import meshlib.mr.Vector3f;

public final class TestAffineXf
{
    static Matrix3f matrix(float[][] rows)
    {
        return new Matrix3f(new Vector3f(rows[0][0], rows[0][1], rows[0][2]), new Vector3f(rows[1][0], rows[1][1], rows[1][2]), new Vector3f(rows[2][0], rows[2][1], rows[2][2]));
    }

    public static void construct()
    {
        AffineXf3f result = new AffineXf3f();
        check(Functions.opEqual(result.A(), Matrix3f.identity()), "identity");
        checkApprox(result.b().x(), 0.0, 0.0001);
        checkApprox(result.b().y(), 0.0, 0.0001);
        checkApprox(result.b().z(), 0.0, 0.0001);
    }

    public static void translation()
    {
        AffineXf3f result = AffineXf3f.translation(new Vector3f(1.0f, 2.0f, 3.0f));
        check(Functions.opEqual(result.A(), Matrix3f.identity()), "identity");
        checkApprox(result.b().x(), 1.0, 0.0001);
        checkApprox(result.b().y(), 2.0, 0.0001);
        checkApprox(result.b().z(), 3.0, 0.0001);
    }

    public static void linear()
    {
        Matrix3f linear = matrix(new float[][]{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
        AffineXf3f result = AffineXf3f.linear(linear);
        check(Functions.opEqual(result.A(), linear), "linear");
        checkApprox(result.b().x(), 0.0, 0.0001);
        checkApprox(result.b().y(), 0.0, 0.0001);
        checkApprox(result.b().z(), 0.0, 0.0001);
    }

    public static void mul()
    {
        AffineXf3f xf1 = AffineXf3f.translation(new Vector3f(1.0f, 2.0f, 3.0f));
        AffineXf3f xf2 = AffineXf3f.linear(matrix(new float[][]{{2, 0, 0}, {0, 2, 0}, {0, 0, 2}}));
        AffineXf3f result = Functions.opMul(xf1, xf2);
        Matrix3f expectedA = Functions.opMul(xf1.A(), xf2.A());
        Matrix3f a = xf1.A();
        Vector3f b2 = xf2.b();
        float ex = a.x().x() * b2.x() + a.y().x() * b2.y() + a.z().x() * b2.z() + xf1.b().x();
        float ey = a.x().y() * b2.x() + a.y().y() * b2.y() + a.z().y() * b2.z() + xf1.b().y();
        float ez = a.x().z() * b2.x() + a.y().z() * b2.y() + a.z().z() * b2.z() + xf1.b().z();
        check(Functions.opEqual(result.A(), expectedA), "A");
        checkApprox(result.b().x(), ex, 0.0001);
        checkApprox(result.b().y(), ey, 0.0001);
        checkApprox(result.b().z(), ez, 0.0001);
    }

    public static void apply()
    {
        AffineXf3f xf = AffineXf3f.translation(new Vector3f(1.0f, 2.0f, 3.0f));
        Vector3f point = new Vector3f(4.0f, 5.0f, 6.0f);
        Vector3f result = xf.call(point);
        Matrix3f a = xf.A();
        float ex = a.x().x() * point.x() + a.y().x() * point.y() + a.z().x() * point.z() + xf.b().x();
        float ey = a.x().y() * point.x() + a.y().y() * point.y() + a.z().y() * point.z() + xf.b().y();
        float ez = a.x().z() * point.x() + a.y().z() * point.y() + a.z().z() * point.z() + xf.b().z();
        checkApprox(result.x(), ex, 0.0001);
        checkApprox(result.y(), ey, 0.0001);
        checkApprox(result.z(), ez, 0.0001);
    }
}
