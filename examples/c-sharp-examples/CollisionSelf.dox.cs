using static MR;

public class CollisionSelf {

    public static void Run(string[] args) {
        var mesh = new MeshPart(MR.makeTorusWithSelfIntersections());

        Console.WriteLine(" --- Beginning Colliding Self Test! --- ");
        var selfCollidingPairs = MR.findSelfCollidingTriangles(mesh);
        if (selfCollidingPairs.isEmpty()) {
            Console.WriteLine("Error with MR.FindSelfCollidingTriangles");
            return;
        }

        FaceFace pair; // more efficient to declare outside loop
        for (ulong i = 0; i < selfCollidingPairs.size(); i++) {
            pair = selfCollidingPairs.at(i);
            Console.WriteLine($"FaceA: {pair.aFace.id} FaceB: {pair.bFace.id}"); // print each pair
        }

        var selfCollidingBitSet = MR.findSelfCollidingTrianglesBS(mesh);
        if (!selfCollidingBitSet.any()) {
            Console.WriteLine("Error with MR.FindSelfCollidingTrianglesBS");
            return;
        }

        Console.WriteLine($"{selfCollidingBitSet.count()} faces self-intersecting");

        var isSelfColliding = !MR.findSelfCollidingTriangles(mesh).isEmpty();
        Console.WriteLine($"Is self colliding: {isSelfColliding}");
    }

}
