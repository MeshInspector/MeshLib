using static MR;

public class CollisionSelfExample
{
    public static void Run(string[] args)
    {
        var mesh = new MeshPart(MR.makeTorusWithSelfIntersections()); // make torus with self-intersections

        // find self-intersecting faces pairs
        var selfCollidingPairs = findSelfCollidingTriangles(mesh);
        for (ulong i = 0; i < selfCollidingPairs.size(); i++)
        {
            var pair = selfCollidingPairs[i];
            Console.WriteLine($"{pair.aFace.id} {pair.bFace.id}"); // print each pair
        }

        // find bitset of self-intersecting faces
        var selfCollidingBitSet = MR.findSelfCollidingTrianglesBS(mesh);
        Console.WriteLine(selfCollidingBitSet.count()); // print number of self-intersecting faces

        // fast check if mesh has self-intersections
        var isSelfColliding = MR.findSelfCollidingTriangles(mesh, outCollidingPairs: null);
        Console.WriteLine(isSelfColliding);
    }
}
