using static MR;

public class CollisionPrecise {

    public static void Run(string[] args) {
        var meshA = MR.MakeUVSphere();
        var meshB = MR.MakeUVSphere();

        meshB.Transform(MR.AffineXf3f.Translation(new MR.Vector3f(0.1f, 0.1f, 0.1f)));

        var meshPartA = new MeshPart(meshA);
        var meshPartB = new MeshPart(meshB);

        Console.WriteLine(" --- Beginning Colliding Precise Test! --- ");
        var converters = MR.GetVectorConverters(meshPartA, meshPartB).ToInt;
        var collidingFaceEdges = MR.FindCollidingEdgeTrisPrecise(meshPartA, meshPartB, converters);

        for (ulong i = 0; i < collidingFaceEdges.Size(); i++) {
            var vet = collidingFaceEdges.At(i);
            var text = vet.IsEdgeATriB()
                ? $"edgeA: {vet.Edge.Id}, triB: {vet.Tri().Id}"
                : $"triA: {vet.Tri().Id}, edgeB: {vet.Edge.Id}";

            Console.WriteLine(text);
        }

        var collidingFaceBitSet = MR.FindCollidingTriangleBitsets(meshPartA, meshPartB);
        var bitSet = collidingFaceBitSet.First();
        Console.WriteLine($"Colliding faces on MeshA: {bitSet.Count()}");
        bitSet = collidingFaceBitSet.Second();
        Console.WriteLine($"Colliding faces on MeshB: {bitSet.Count()}");

        var isColliding = !MR.FindCollidingTriangles(meshPartA, meshPartB, null, true).IsEmpty();
        Console.WriteLine($"Is Colliding: {isColliding}\n");

    }

}

