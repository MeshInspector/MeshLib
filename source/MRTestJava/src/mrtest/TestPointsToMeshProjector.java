package mrtest;

import static mrtest.Common.checkApprox;
import static mrtest.Common.checkEq;

import meshlib.mr.AffineXf3f;
import meshlib.mr.Functions;
import meshlib.mr.Mesh;
import meshlib.mr.MeshProjectionParameters;
import meshlib.mr.Vector3f;
import meshlib.mr.VertId;
import meshlib.mr.VertScalars;

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
