using static MR;

public class CollisionExample
{
    public static void Run(string[] args)
    {
        var meshA = MR.makeUVSphere(); // make mesh A
        var meshB = MR.makeUVSphere(); // make mesh B
        meshB.transform(MR.AffineXf3f.translation(new MR.Vector3f(0.1f, 0.1f, 0.1f))); // shift mesh B for better demonstration

        var meshPartA = new MeshPart(meshA);
        var meshPartB = new MeshPart(meshB);

        var collidingFacePairs = MR.findCollidingTriangles(meshPartA, meshPartB); // find each pair of colliding faces
        for (ulong i = 0; i < collidingFacePairs.size(); i++)
        {
            var pair = collidingFacePairs[i];
            Console.WriteLine($"{pair.aFace.id} {pair.bFace.id}"); // print each pair of colliding faces
        }

        var collidingFaceBitSets = MR.findCollidingTriangleBitsets(meshPartA, meshPartB); // find bitsets of colliding faces
        Console.WriteLine(collidingFaceBitSets.first().count()); // print number of colliding faces from mesh A
        Console.WriteLine(collidingFaceBitSets.second().count()); // print number of colliding faces from mesh B

        var isColliding = !MR.findCollidingTriangles(meshPartA, meshPartB, null, true).empty(); // fast check if mesh A and mesh B collide
        Console.WriteLine(isColliding);
    }
}
