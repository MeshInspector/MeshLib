package mrtest;

import static mrtest.Common.check;

import meshlib.mr.AffineXf3f;
import meshlib.mr.Functions;
import meshlib.mr.Mesh;
import meshlib.mr.MeshMeshDistanceResult;
import meshlib.mr.MeshPart;
import meshlib.mr.Vector3f;

public final class TestMeshMeshDistance
{
    public static void meshMeshDistance()
    {
        Mesh sphere1 = Functions.makeUVSphere(1.0f, 8, 8);
        MeshPart wholeSphere1 = new MeshPart(sphere1, null);
        MeshMeshDistanceResult d11 = Functions.findDistance(wholeSphere1, wholeSphere1, null, Float.MAX_VALUE);
        check(d11.distSq() == 0.0f, "self distance");

        AffineXf3f zShift = AffineXf3f.translation(new Vector3f(0.0f, 0.0f, 3.0f));
        MeshMeshDistanceResult d1z = Functions.findDistance(wholeSphere1, wholeSphere1, zShift, Float.MAX_VALUE);
        check(d1z.distSq() == 1.0f, "shifted distance");

        Mesh sphere2 = Functions.makeUVSphere(2.0f, 8, 8);
        MeshPart wholeSphere2 = new MeshPart(sphere2, null);
        MeshMeshDistanceResult d12 = Functions.findDistance(wholeSphere1, wholeSphere2, null, Float.MAX_VALUE);
        double dist12 = Math.sqrt(d12.distSq());
        check(dist12 > 0.9 && dist12 < 1.0, "distance 12");
    }
}
