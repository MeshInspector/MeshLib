using static MR;

public class CollisionSelf {

    public static void Run(string[] args) {
        var mesh = new MeshPart(MR.MakeTorusWithSelfIntersections());

        Console.WriteLine(" --- Beginning Colliding Self Test! --- ");
        var selfCollidingPairs = MR.FindSelfCollidingTriangles(mesh);
        if (selfCollidingPairs.IsEmpty()) {
            Console.WriteLine("Error with MR.FindSelfCollidingTriangles");
            return;
        }

        FaceFace pair; // more efficient to declare outside loop
        for (ulong i = 0; i < selfCollidingPairs.Size(); i++) {
            pair = selfCollidingPairs.At(i);
            Console.WriteLine($"FaceA: {pair.AFace.Id} FaceB: {pair.BFace.Id}"); // print each pair
        }

        var selfCollidingBitSet = MR.FindSelfCollidingTrianglesBS(mesh);
        if (!selfCollidingBitSet.Any()) {
            Console.WriteLine("Error with MR.FindSelfCollidingTrianglesBS");
            return;
        }

        Console.WriteLine($"{selfCollidingBitSet.Count()} faces self-intersecting");

        var isSelfColliding = !MR.FindSelfCollidingTriangles(mesh).IsEmpty();
        Console.WriteLine($"Is self colliding: {isSelfColliding}");
    }

}

