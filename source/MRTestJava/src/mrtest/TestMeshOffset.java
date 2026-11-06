package mrtest;

import static mrtest.Common.check;

import meshlib.Functions;
import meshlib.GeneralOffsetParameters;
import meshlib.Mesh;
import meshlib.MeshPart;
import meshlib.OffsetMode;
import meshlib.OffsetParameters;
import meshlib.PartMapping;
import meshlib.SharpOffsetParameters;

public final class TestMeshOffset
{
    public static void offsetMesh()
    {
        Mesh mesh = Common.createCube();
        MeshPart mp = new MeshPart(mesh, null);
        OffsetParameters params = new OffsetParameters();
        params.setVoxelSize(Functions.suggestVoxelSize(mp, 10000.0f));
        check(Functions.offsetMesh(mp, 0.1f, params) != null, "offsetMesh");
    }

    public static void doubleOffsetMesh()
    {
        Mesh mesh = Common.createCube();
        MeshPart mp = new MeshPart(mesh, null);
        OffsetParameters params = new OffsetParameters();
        params.setVoxelSize(Functions.suggestVoxelSize(mp, 10000.0f));
        check(Functions.doubleOffsetMesh(mp, 0.1f, -0.2f, params) != null, "doubleOffsetMesh");
    }

    public static void mcOffsetMesh()
    {
        Mesh mesh = Common.createCube();
        MeshPart mp = new MeshPart(mesh, null);
        OffsetParameters params = new OffsetParameters();
        params.setVoxelSize(Functions.suggestVoxelSize(mp, 10000.0f));
        check(Functions.mcOffsetMesh(mp, 0.1f, params, null) != null, "mcOffsetMesh");
    }

    public static void sharpOffsetMesh()
    {
        Mesh mesh = Common.createCube();
        MeshPart mp = new MeshPart(mesh, null);
        SharpOffsetParameters params = new SharpOffsetParameters();
        params.setVoxelSize(Functions.suggestVoxelSize(mp, 10000.0f));
        check(Functions.sharpOffsetMesh(mp, 0.1f, params) != null, "sharpOffsetMesh");
    }

    public static void generalOffsetMesh()
    {
        Mesh mesh = Common.createCube();
        MeshPart mp = new MeshPart(mesh, null);
        GeneralOffsetParameters params = new GeneralOffsetParameters();
        params.setVoxelSize(Functions.suggestVoxelSize(mp, 10000.0f));
        params.setMode(OffsetMode.Standard);
        check(Functions.generalOffsetMesh(mp, 0.1f, params) != null, "generalOffsetMesh");
    }

    public static void thickenMesh()
    {
        Mesh mesh = Common.createCube();
        MeshPart mp = new MeshPart(mesh, null);
        GeneralOffsetParameters params = new GeneralOffsetParameters();
        params.setVoxelSize(Functions.suggestVoxelSize(mp, 10000.0f));
        params.setMode(OffsetMode.Standard);
        PartMapping map = new PartMapping();
        check(Functions.thickenMesh(mesh, 0.1f, params, map) != null, "thickenMesh");
    }
}
