public static class CollisionExample
{
    public static void Main(string[] args)
    {
        var meshA = MR.makeUVSphere(); // make mesh A
        var meshB = MR.makeUVSphere(); // make mesh B
        meshB.transform(MR.AffineXf3f.translation(new MR.Vector3f(0.1f, 0.1f, 0.1f))); // shift mesh B for better demonstration

        var collidingFacePairs = MR.findCollidingTriangles(meshA, meshB); // find each pair of colliding faces
        for (nuint i = 0; i < collidingFacePairs.size(); ++i)
            Console.WriteLine("{0} {1}", collidingFacePairs[i].aFace.id, collidingFacePairs[i].bFace.id); // print each pair of colliding faces

        var collidingFaceBitSets = MR.findCollidingTriangleBitsets(meshA, meshB); // find bitsets of colliding faces
        Console.WriteLine(collidingFaceBitSets.first().count()); // print number of colliding faces from mesh A
        Console.WriteLine(collidingFaceBitSets.second().count()); // print number of colliding faces from mesh B

        var isColliding = !MR.findCollidingTriangles(meshA, meshB, null, true).empty(); // fast check if mesh A and mesh B collide
        Console.WriteLine(isColliding);
    }
}
