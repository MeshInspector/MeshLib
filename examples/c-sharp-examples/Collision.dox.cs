using static MR;

public class Collision {

    public static void Run(string[] args) {

        var meshA = MR.MakeUVSphere();
        var meshB = MR.MakeUVSphere();

        meshB.Transform(MR.AffineXf3f.Translation(new MR.Vector3f(0.1f, 0.1f, 0.1f)));

        var meshPartA = new MeshPart(meshA);
        var meshPartB = new MeshPart(meshB);

        Console.WriteLine(" --- Beginning Colliding Test! --- ");
        var collidingFacePairs = MR.FindCollidingTriangles(meshPartA, meshPartB);

        for (ulong i = 0; i < collidingFacePairs.Size(); i++) {
            var pair = collidingFacePairs.At(i);
            Console.WriteLine($"FaceA: {pair.AFace.Id} FaceB: {pair.BFace.Id}");
        }

        var collidingFaceBitSet = MR.FindCollidingTriangleBitsets(meshPartA, meshPartB);
        var bitSet = collidingFaceBitSet.First();
        Console.WriteLine($"Colliding faces on MeshA: {bitSet.Count()}");
        bitSet = collidingFaceBitSet.Second();
        Console.WriteLine($"Colliding faces on MeshB: {bitSet.Count()}");

        var isColliding = !MR.FindCollidingTriangles(meshPartA, meshPartB, null, true).IsEmpty();
        Console.WriteLine($"Meshes are colliding: {isColliding}\n");
    }
}
