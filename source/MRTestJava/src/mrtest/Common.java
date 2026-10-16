package mrtest;

import meshlib.mr.Functions;
import meshlib.mr.Mesh;
import meshlib.mr.Vector3f;

final class Common
{
    private Common() {}

    static final float PI_F = 3.14159265358979f;

    /** Creates a cube mesh with diagonal = 1. */
    static Mesh createCube()
    {
        return Functions.makeCube(Vector3f.diagonal(1.0f), Vector3f.diagonal(-0.5f));
    }

    /** Creates a torus with the given radii and resolutions. */
    static Mesh makeTorus(float primaryRadius, float secondaryRadius, int primaryResolution, int secondaryResolution)
    {
        return Functions.makeTorus(primaryRadius, secondaryRadius, primaryResolution, secondaryResolution, null);
    }

    static void check(boolean condition, String what)
    {
        if (!condition)
            throw new AssertionError(what);
    }

    static void checkEq(long actual, long expected, String what)
    {
        if (actual != expected)
            throw new AssertionError(what + ": expected " + expected + ", got " + actual);
    }

    static void checkApprox(double actual, double expected, double eps)
    {
        if (!(Math.abs(actual - expected) <= eps))
            throw new AssertionError("expected " + expected + " (delta " + eps + "), got " + actual + " (delta " + Math.abs(actual - expected) + ")");
    }
}
