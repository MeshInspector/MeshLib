using static MR;

public class CollisionPreciseExample
{
    public static void Run(string[] args)
    {
        var meshA = MR.makeUVSphere(); // make mesh A
        var meshB = MR.makeUVSphere(); // make mesh B
        meshB.transform(MR.AffineXf3f.translation(new MR.Vector3f(0.1f, 0.1f, 0.1f))); // shift mesh B for better demonstration

        var meshPartA = new MeshPart(meshA);
        var meshPartB = new MeshPart(meshB);

        var converters = MR.getVectorConverters(meshPartA, meshPartB).toInt; // create converters to integer field (needed for absolute precision predicates)
        var collidingFaceEdges = MR.findCollidingEdgeTrisPrecise(meshPartA, meshPartB, converters); // find each intersecting edge/triangle pair
        // print pairs of edges triangles
        for (ulong i = 0; i < collidingFaceEdges.size(); i++)
        {
            var vet = collidingFaceEdges[i];
            if (vet.isEdgeATriB())
                Console.WriteLine($"edgeA: {vet.edge.id}, triB: {vet.tri().id}");
            else
                Console.WriteLine($"triA: {vet.tri().id}, edgeB: {vet.edge.id}");
        }
    }
}
