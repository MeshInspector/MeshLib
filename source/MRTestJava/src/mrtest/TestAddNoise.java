package mrtest;

import meshlib.mr.Functions;
import meshlib.mr.Mesh;

public final class TestAddNoise
{
    public static void addNoise()
    {
        Mesh mesh = Common.makeTorus(1.1f, 0.5f, 16, 16);
        Functions.addNoise(mesh, null, null);
    }
}
