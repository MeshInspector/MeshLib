public static class CollisionPreciseExample
{
    public static void Main(string[] args)
    {
        var meshA = MR.makeUVSphere(); // make mesh A
        var meshB = MR.makeUVSphere(); // make mesh B
        meshB.transform(MR.AffineXf3f.translation(new MR.Vector3f(0.1f, 0.1f, 0.1f))); // shift mesh B for better demonstration

        var converters = MR.getVectorConverters(meshA, meshB); // create converters to integer field (needed for absolute precision predicates)
        var collidingFaceEdges = MR.findCollidingEdgeTrisPrecise(meshA, meshB, converters.toInt); // find each intersecting edge/triangle pair
        // print pairs of edges triangles
        for (nuint i = 0; i < collidingFaceEdges.size(); ++i)
        {
            var vet = collidingFaceEdges[i];
            if (vet.isEdgeATriB())
                Console.WriteLine("edgeA: {0}, triB: {1}", vet.edge.id, vet.tri().id);
            else
                Console.WriteLine("triA: {0}, edgeB: {1}", vet.tri().id, vet.edge.id);
        }
    }
}
