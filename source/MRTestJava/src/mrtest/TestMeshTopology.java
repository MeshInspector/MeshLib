package mrtest;

import static mrtest.Common.check;
import static mrtest.Common.checkEq;

import meshlib.mr.EdgeId;
import meshlib.mr.Mesh;
import meshlib.mr.VertId;

public final class TestMeshTopology
{
    public static void pack()
    {
        Mesh mesh = Common.createCube();
        mesh.topology().pack(null, null, null, null);
    }

    public static void getValidVerts()
    {
        Mesh mesh = Common.createCube();
        checkEq(mesh.topology().getValidVerts().count(), 8, "valid verts");
    }

    public static void getValidFaces()
    {
        Mesh mesh = Common.createCube();
        checkEq(mesh.topology().getValidFaces().count(), 12, "valid faces");
    }

    public static void findHoleRepresentiveEdges()
    {
        Mesh mesh = Common.createCube();
        checkEq(mesh.topology().findHoleRepresentiveEdges(null).size(), 0, "holes");
    }

    public static void getLeftTriVerts()
    {
        Mesh mesh = Common.createCube();
        VertId v0 = new VertId(-1), v1 = new VertId(-1), v2 = new VertId(-1);
        mesh.topology().getLeftTriVerts(new EdgeId(0), v0, v1, v2);
        check(v0.id_() >= 0 && v1.id_() >= 0 && v2.id_() >= 0, "verts");
    }

    public static void findNumHoles()
    {
        Mesh mesh = Common.createCube();
        checkEq(mesh.topology().findNumHoles(null), 0, "holes");
    }

    public static void faceSize()
    {
        Mesh mesh = Common.createCube();
        checkEq(mesh.topology().faceSize(), 12, "face size");
    }

    public static void getTriangulation()
    {
        Mesh mesh = Common.createCube();
        checkEq(mesh.topology().getTriangulation().size(), 12, "triangulation");
    }
}
