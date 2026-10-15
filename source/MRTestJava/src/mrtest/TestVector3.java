package mrtest;

import static mrtest.Common.check;
import static mrtest.Common.checkApprox;

import meshlib.mr.Functions;
import meshlib.mr.Vector3f;
import meshlib.mr.Vector3i;

public final class TestVector3
{
    static void checkVec(Vector3f v, float x, float y, float z)
    {
        checkApprox(v.x(), x, 0.0001);
        checkApprox(v.y(), y, 0.0001);
        checkApprox(v.z(), z, 0.0001);
    }

    public static void diagonal()
    {
        checkVec(Vector3f.diagonal(3.0f), 3.0f, 3.0f, 3.0f);
    }

    public static void plusX()
    {
        checkVec(Vector3f.plusX(), 1.0f, 0.0f, 0.0f);
    }

    public static void plusY()
    {
        checkVec(Vector3f.plusY(), 0.0f, 1.0f, 0.0f);
    }

    public static void plusZ()
    {
        checkVec(Vector3f.plusZ(), 0.0f, 0.0f, 1.0f);
    }

    public static void add()
    {
        checkVec(Functions.opAdd(new Vector3f(1.0f, 2.0f, 3.0f), new Vector3f(4.0f, 5.0f, 6.0f)), 5.0f, 7.0f, 9.0f);
    }

    public static void sub()
    {
        checkVec(Functions.opSub(new Vector3f(5.0f, 7.0f, 9.0f), new Vector3f(4.0f, 5.0f, 6.0f)), 1.0f, 2.0f, 3.0f);
    }

    public static void mulScalar()
    {
        checkVec(Functions.opMul(new Vector3f(1.0f, 2.0f, 3.0f), 2.0f), 2.0f, 4.0f, 6.0f);
    }

    public static void lengthSq()
    {
        checkApprox(new Vector3f(1.0f, 2.0f, 3.0f).lengthSq(), 14.0, 0.0001);
    }

    public static void length()
    {
        checkApprox(new Vector3f(3.0f, 4.0f, 0.0f).length(), 5.0, 0.0001);
    }

    public static void diagonalI()
    {
        Vector3i v = Vector3i.diagonal(3);
        check(v.x() == 3 && v.y() == 3 && v.z() == 3, "diagonal");
    }

    public static void plusXI()
    {
        Vector3i v = Vector3i.plusX();
        check(v.x() == 1 && v.y() == 0 && v.z() == 0, "plusX");
    }

    public static void plusYI()
    {
        Vector3i v = Vector3i.plusY();
        check(v.x() == 0 && v.y() == 1 && v.z() == 0, "plusY");
    }

    public static void plusZI()
    {
        Vector3i v = Vector3i.plusZ();
        check(v.x() == 0 && v.y() == 0 && v.z() == 1, "plusZ");
    }
}
