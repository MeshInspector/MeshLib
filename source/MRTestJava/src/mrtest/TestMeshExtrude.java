package mrtest;

import static mrtest.Common.check;
import static mrtest.Common.checkEq;

import meshlib.mr.FaceBitSet;
import meshlib.mr.FaceId;
import meshlib.mr.Functions;
import meshlib.mr.Mesh;

public final class TestMeshExtrude
{
    // The region contains selected faces.
    public static void degenerateBandNonEmpty()
    {
        Mesh mesh = Common.makeTorus(1.1f, 0.5f, 16, 16);
        long countBefore = mesh.points().size();
        FaceBitSet region = new FaceBitSet();
        region.autoResizeSet(new FaceId(0), true);
        region.autoResizeSet(new FaceId(1), true);
        Functions.makeDegenerateBandAroundRegion(mesh, region, null);
        mesh.invalidateCaches(true);
        // The number of vertices grows because of the vertex duplication along the boundary.
        check(mesh.points().size() != countBefore, "points added");
    }

    // The region is empty.
    public static void degenerateBandEmpty()
    {
        Mesh mesh = Common.makeTorus(1.1f, 0.5f, 16, 16);
        long countBefore = mesh.points().size();
        FaceBitSet region = new FaceBitSet();
        Functions.makeDegenerateBandAroundRegion(mesh, region, null);
        mesh.invalidateCaches(true);
        checkEq(mesh.points().size(), countBefore, "points unchanged");
    }
}
