public static class CollisionSelfExample
{
    public static void Main(string[] args)
    {
        var mesh = MR.makeTorusWithSelfIntersections(); // make torus with self-intersections

        var selfCollidingPairs = MR.findSelfCollidingTriangles(mesh); // find self-intersecting faces pairs
        for (nuint i = 0; i < selfCollidingPairs.size(); ++i)
            Console.WriteLine("{0} {1}", selfCollidingPairs[i].aFace.id, selfCollidingPairs[i].bFace.id); // print each pair

        var selfCollidingBitSet = MR.findSelfCollidingTrianglesBS(mesh); // find bitset of self-intersecting faces
        Console.WriteLine(selfCollidingBitSet.count()); // print number of self-intersecting faces

        var isSelfColliding = MR.findSelfCollidingTriangles(mesh, outCollidingPairs: null); // fast check if mesh has self-intersections
        Console.WriteLine(isSelfColliding);
    }
}
