package mrtest;

import static mrtest.Common.checkEq;

import meshlib.mr.Functions;
import meshlib.mr.Mesh;
import meshlib.mr.SphereParams;
import meshlib.mr.VertId;
import meshlib.mr.VertMap;

public final class TestMeshBuilder
{
    public static void uniteCloseVertices()
    {
        SphereParams params = new SphereParams();
        params.setRadius(1.0f);
        params.setNumMeshVertices(3000);
        Mesh mesh = Functions.makeSphere(params);

        VertMap vertMap = new VertMap();
        int united = meshlib.mr.meshbuilder.Functions.uniteCloseVertices(mesh, 0.1f, false, vertMap);
        checkEq(united, 2230, "united");
        checkEq(vertMap.index(new VertId(1000)).id_(), 42, "map");
    }
}
