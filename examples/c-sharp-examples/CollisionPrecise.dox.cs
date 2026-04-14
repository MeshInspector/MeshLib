using static MR;

public class CollisionPrecise {

    public static void Run(string[] args) {
        var meshA = MR.makeUVSphere();
        var meshB = MR.makeUVSphere();

        meshB.transform(MR.AffineXf3f.translation(new MR.Vector3f(0.1f, 0.1f, 0.1f)));

        var meshPartA = new MeshPart(meshA);
        var meshPartB = new MeshPart(meshB);

        Console.WriteLine(" --- Beginning Colliding Precise Test! --- ");
        var converters = MR.getVectorConverters(meshPartA, meshPartB).toInt;
        var collidingFaceEdges = MR.findCollidingEdgeTrisPrecise(meshPartA, meshPartB, converters);

        for (ulong i = 0; i < collidingFaceEdges.size(); i++) {
            var vet = collidingFaceEdges[i];
            var text = vet.isEdgeATriB()
                ? $"edgeA: {vet.edge.id}, triB: {vet.tri().id}"
                : $"triA: {vet.tri().id}, edgeB: {vet.edge.id}";

            Console.WriteLine(text);
        }

        var collidingFaceBitSet = MR.findCollidingTriangleBitsets(meshPartA, meshPartB);
        var bitSet = collidingFaceBitSet.first();
        Console.WriteLine($"Colliding faces on MeshA: {bitSet.count()}");
        bitSet = collidingFaceBitSet.second();
        Console.WriteLine($"Colliding faces on MeshB: {bitSet.count()}");

        var isColliding = !MR.findCollidingTriangles(meshPartA, meshPartB, null, true).empty();
        Console.WriteLine($"Is Colliding: {isColliding}\n");

    }

}
