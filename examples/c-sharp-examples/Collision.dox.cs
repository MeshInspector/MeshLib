using static MR;

public class Collision {

    public static void Run(string[] args) {

        var meshA = MR.makeUVSphere();
        var meshB = MR.makeUVSphere();

        meshB.transform(MR.AffineXf3f.translation(new MR.Vector3f(0.1f, 0.1f, 0.1f)));

        var meshPartA = new MeshPart(meshA);
        var meshPartB = new MeshPart(meshB);

        Console.WriteLine(" --- Beginning Colliding Test! --- ");
        var collidingFacePairs = MR.findCollidingTriangles(meshPartA, meshPartB);

        for (ulong i = 0; i < collidingFacePairs.size(); i++) {
            var pair = collidingFacePairs.at(i);
            Console.WriteLine($"FaceA: {pair.aFace.id} FaceB: {pair.bFace.id}");
        }

        var collidingFaceBitSet = MR.findCollidingTriangleBitsets(meshPartA, meshPartB);
        var bitSet = collidingFaceBitSet.first();
        Console.WriteLine($"Colliding faces on MeshA: {bitSet.count()}");
        bitSet = collidingFaceBitSet.second();
        Console.WriteLine($"Colliding faces on MeshB: {bitSet.count()}");

        var isColliding = !MR.findCollidingTriangles(meshPartA, meshPartB, null, true).isEmpty();
        Console.WriteLine($"Meshes are colliding: {isColliding}\n");
    }
}
