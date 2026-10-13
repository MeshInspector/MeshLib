package mrtest;

import static mrtest.Common.checkApprox;

import meshlib.mr.FaceBitSet;
import meshlib.mr.Functions;
import meshlib.mr.Mesh;
import meshlib.mr.SphereParams;
import meshlib.mr.Vector3f;

public final class TestSimpleObjects
{
    static double area(Mesh mesh)
    {
        return mesh.area((FaceBitSet)null);
    }

    public static void makeCube()
    {
        Mesh cube = Functions.makeCube(new Vector3f(1.0f, 1.0f, 1.0f), new Vector3f(0.0f, 0.0f, 0.0f));
        checkApprox(area(cube), 6.0, 0.001);
    }

    public static void makeCylinderAdvanced()
    {
        Mesh cylinder = Functions.makeCylinderAdvanced(1.0f, 0.5f, 0.0f, 2.0f * 3.1415f, 1.0f, 16);
        checkApprox(area(cylinder), 10.541437, 0.1);
    }

    public static void makeTorus()
    {
        Mesh torus = Functions.makeTorus(2.0f, 0.5f, 32, 32, null);
        checkApprox(area(torus), 39.257, 0.1);
    }

    public static void makeTorusWithSelfIntersections()
    {
        Mesh torus = Functions.makeTorusWithSelfIntersections(2.0f, 0.5f, 16, 16, null);
        checkApprox(area(torus), 57.583, 0.01);
    }

    public static void makeSphere()
    {
        SphereParams params = new SphereParams();
        params.setRadius(1.0f);
        params.setNumMeshVertices(100);
        Mesh sphere = Functions.makeSphere(params);
        checkApprox(area(sphere), 12.157, 0.1);
    }

    public static void makeUVSphere()
    {
        Mesh sphere = Functions.makeUVSphere(1.5f, 32, 32);
        checkApprox(area(sphere), 28.15, 0.1);
    }
}
