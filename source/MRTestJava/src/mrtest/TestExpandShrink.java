package mrtest;

import static mrtest.Common.check;
import static mrtest.Common.checkEq;

import meshlib.mr.FaceBitSet;
import meshlib.mr.FaceId;
import meshlib.mr.Functions;
import meshlib.mr.Mesh;
import meshlib.mr.MeshTopology;
import meshlib.mr.SphereParams;
import meshlib.mr.VertBitSet;
import meshlib.mr.VertId;

public final class TestExpandShrink
{
    static Mesh makeSphere()
    {
        SphereParams params = new SphereParams();
        params.setNumMeshVertices(3000);
        params.setRadius(1.0f);
        return Functions.makeSphere(params);
    }

    public static void expandShrink()
    {
        Mesh mesh = makeSphere();
        MeshTopology top = mesh.topology();
        FaceBitSet region = Functions.expand(top, new FaceId(0), 3);
        checkEq(region.count(), 75, "expand from face");
        Functions.expand(top, region, 3);
        long num = region.count();
        check(num == 274 || // without FMA instruction (default settings for x86 or old compilers for ARM)
              num == 284, // with FMA instruction (modern compilers for ARM)
              "expand region: " + num);
        Functions.shrink(top, region, 3);
        checkEq(region.count(), 75, "shrink");
    }

    public static void expandShrinkVerts()
    {
        Mesh mesh = makeSphere();
        MeshTopology top = mesh.topology();
        VertBitSet region = Functions.expand(top, new VertId(0), 3);
        checkEq(region.count(), 37, "expand from vert");
        Functions.expand(top, region, 3);
        check(region.count() > 37, "expand region"); // platform dependent results
        Functions.shrink(top, region, 3);
        checkEq(region.count(), 37, "shrink");
    }
}
