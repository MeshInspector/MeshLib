using static MR;

public class CollisionSelf
{
    public static void Run(string[] args)
    {
        var mesh = new MeshPart(MR.makeTorusWithSelfIntersections());

        Console.WriteLine(" --- Beginning Colliding Self Test! --- ");
        // find self-intersecting faces pairs
        var selfCollidingPairs = findSelfCollidingTriangles(mesh);

        FaceFace pair; // more efficient to declare outside loop
        for (ulong i = 0; i < selfCollidingPairs.size(); i++)
        {
            pair = selfCollidingPairs[i];
            Console.WriteLine($"FaceA: {pair.aFace.id} FaceB: {pair.bFace.id}"); // print each pair
        }

        // find bitset of self-intersecting faces
        var selfCollidingBitSet = MR.findSelfCollidingTrianglesBS(mesh);
        Console.WriteLine($"{selfCollidingBitSet.count()} faces self-intersecting");

        // fast check if mesh has self-intersections
        var isSelfColliding = MR.findSelfCollidingTriangles(mesh, outCollidingPairs: null);
        Console.WriteLine($"Is self colliding: {isSelfColliding}");
    }

}
