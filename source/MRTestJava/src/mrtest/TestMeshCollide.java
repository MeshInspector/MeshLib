package mrtest;

import static mrtest.Common.check;

import meshlib.mr.Functions;
import meshlib.mr.Mesh;
import meshlib.mr.MeshPart;

public final class TestMeshCollide
{
    public static void meshCollide()
    {
        Mesh meshA = Common.makeTorus(1.1f, 0.5f, 8, 8);
        Mesh meshB = Common.makeTorus(1.1f, 0.2f, 8, 8);
        MeshPart partA = new MeshPart(meshA, null);
        MeshPart partB = new MeshPart(meshB, null);
        check(Functions.isInside(partB, partA, null), "B inside A");
        check(!Functions.isInside(partA, partB, null), "A not inside B");
    }
}
