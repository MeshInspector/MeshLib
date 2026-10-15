package mrtest;

import static mrtest.Common.check;
import static mrtest.Common.checkApprox;

import meshlib.mr.Functions;
import meshlib.mr.Matrix3f;
import meshlib.mr.Vector3f;

public final class TestMatrix3
{
    static Matrix3f matrix(float[][] rows)
    {
        return new Matrix3f(new Vector3f(rows[0][0], rows[0][1], rows[0][2]), new Vector3f(rows[1][0], rows[1][1], rows[1][2]), new Vector3f(rows[2][0], rows[2][1], rows[2][2]));
    }

    static void checkMatrix(Matrix3f m, float[][] rows)
    {
        float[][] actual = {{m.x().x(), m.x().y(), m.x().z()}, {m.y().x(), m.y().y(), m.y().z()}, {m.z().x(), m.z().y(), m.z().z()}};
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                checkApprox(actual[i][j], rows[i][j], 0.001);
    }

    public static void identity()
    {
        checkMatrix(Matrix3f.identity(), new float[][]{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}});
    }

    public static void rotationScalar()
    {
        Matrix3f m = Matrix3f.rotation(new Vector3f(1.0f, 0.0f, 0.0f), 3.14159f / 2.0f); // 90 degrees around the X axis
        checkApprox(m.x().x(), 1.0, 0.001);
        checkApprox(m.y().y(), 0.0, 0.001);
        checkApprox(m.y().z(), -1.0, 0.001);
        checkApprox(m.z().y(), 1.0, 0.001);
        checkApprox(m.z().z(), 0.0, 0.001);
    }

    public static void rotationVector()
    {
        Matrix3f m = Matrix3f.rotation(new Vector3f(1.0f, 0.0f, 0.0f), new Vector3f(0.0f, 1.0f, 0.0f));
        checkMatrix(m, new float[][]{{0, -1, 0}, {1, 0, 0}, {0, 0, 1}});
    }

    public static void add()
    {
        Matrix3f a = matrix(new float[][]{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
        Matrix3f b = matrix(new float[][]{{9, 8, 7}, {6, 5, 4}, {3, 2, 1}});
        checkMatrix(Functions.opAdd(a, b), new float[][]{{10, 10, 10}, {10, 10, 10}, {10, 10, 10}});
    }

    public static void sub()
    {
        Matrix3f a = matrix(new float[][]{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
        Matrix3f b = matrix(new float[][]{{9, 8, 7}, {6, 5, 4}, {3, 2, 1}});
        checkMatrix(Functions.opSub(a, b), new float[][]{{-8, -6, -4}, {-2, 0, 2}, {4, 6, 8}});
    }

    public static void mul()
    {
        Matrix3f a = Matrix3f.identity();
        Matrix3f b = matrix(new float[][]{{2, 1, 0}, {1, 2, 0}, {0, 0, 3}});
        checkMatrix(Functions.opMul(a, b), new float[][]{{2, 1, 0}, {1, 2, 0}, {0, 0, 3}});
    }

    public static void mulVector()
    {
        Matrix3f m = matrix(new float[][]{{2, 0, 0}, {0, 2, 0}, {0, 0, 2}});
        Vector3f r = Functions.opMul(m, new Vector3f(1.0f, 2.0f, 3.0f));
        checkApprox(r.x(), 2.0, 0.001);
        checkApprox(r.y(), 4.0, 0.001);
        checkApprox(r.z(), 6.0, 0.001);
    }

    public static void equal()
    {
        Matrix3f a = Matrix3f.identity();
        Matrix3f b = Matrix3f.identity();
        Matrix3f c = matrix(new float[][]{{2, 0, 0}, {0, 2, 0}, {0, 0, 2}});
        check(Functions.opEqual(a, b), "equal");
        check(!Functions.opEqual(a, c), "not equal");
    }
}
