package mrtest;

import static mrtest.Common.checkEq;

import meshlib.Functions;
import meshlib.Mesh;
import meshlib.SphereParams;
import meshlib.VertId;
import meshlib.VertMap;

public final class TestMeshBuilder
{
    public static void uniteCloseVertices()
    {
        SphereParams params = new SphereParams();
        params.setRadius(1.0f);
        params.setNumMeshVertices(3000);
        Mesh mesh = Functions.makeSphere(params);

        VertMap vertMap = new VertMap();
        int united = meshlib.meshbuilder.Functions.uniteCloseVertices(mesh, 0.1f, false, vertMap);
        checkEq(united, 2230, "united");
        checkEq(vertMap.index(new VertId(1000)).id_(), 42, "map");
    }
}
