package mrtest;

import static mrtest.Common.checkEq;

import meshlib.mr.Functions;
import meshlib.mr.Mesh;

public final class TestMeshNormals
{
    public static void meshNormals()
    {
        Mesh cube = Common.createCube();
        checkEq(Functions.computePerVertNormals(cube).size(), 8, "vert normals");
        checkEq(Functions.computePerFaceNormals(cube).size(), 12, "face normals");
    }
}
