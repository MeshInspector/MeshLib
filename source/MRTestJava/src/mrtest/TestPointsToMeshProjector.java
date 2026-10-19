package mrtest;

import static mrtest.Common.checkApprox;
import static mrtest.Common.checkEq;

import meshlib.AffineXf3f;
import meshlib.Functions;
import meshlib.Mesh;
import meshlib.MeshProjectionParameters;
import meshlib.Vector3f;
import meshlib.VertId;
import meshlib.VertScalars;

public final class TestPointsToMeshProjector
{
    public static void findSignedDistances()
    {
        Mesh meshA = Common.createCube();
        Mesh meshB = Common.createCube();
        Vector3f shift = new Vector3f(1.0f, 2.0f, 3.0f);
        AffineXf3f xf = AffineXf3f.translation(shift);
        MeshProjectionParameters params = new MeshProjectionParameters();
        params.setXf(xf);
        VertScalars results = Functions.findSignedDistances(meshA, meshB, params, null);
        checkEq(results.size(), meshB.points().size(), "size");
        float maxDist = 0.0f;
        for (long i = 0; i < results.size(); i++)
            maxDist = Math.max(maxDist, Math.abs(results.index(new VertId((int)i))));
        checkApprox(maxDist, shift.length(), 1e-6);
    }
}
