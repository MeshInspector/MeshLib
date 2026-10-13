package mrtest;

import static mrtest.Common.check;
import static mrtest.Common.checkEq;

import meshlib.mr.Functions;
import meshlib.mr.Mesh;
import meshlib.mr.MeshTopology;
import meshlib.mr.PointCloud;
import meshlib.mr.VertNormalsMode;
import meshlib.mr.std_optional_MR_Mesh;

public final class TestPointCloud
{
    public static void triangulation()
    {
        Mesh mesh = Common.makeTorus(2.0f, 1.0f, 32, 32);
        PointCloud pc = Functions.meshToPointCloud(mesh, VertNormalsMode.AreaWeighted, null);
        std_optional_MR_Mesh restored = Functions.triangulatePointCloud(pc, null, null);
        check(restored.has_value(), "triangulatePointCloud failed");
        checkEq(restored.value().points().size(), 1024, "points");
        MeshTopology top = restored.value().topology();
        checkEq(top.getValidVerts().count(), 1024, "valid verts");
        checkEq(top.findHoleRepresentiveEdges(null).size(), 0, "holes");
    }
}
