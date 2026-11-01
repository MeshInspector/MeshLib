import meshlib.FaceBitSet;
import meshlib.FaceFace;
import meshlib.Functions;
import meshlib.Mesh;
import meshlib.MeshPart;
import meshlib.std_vector_MR_FaceFace;

// The class isn't public because of the `.dox.java` file name, the launcher doesn't mind.
class CollisionSelf
{
    public static void main(String[] args)
    {
        Mesh mesh = Functions.makeTorusWithSelfIntersections(null, null, null, null, null); // make torus with self-intersections
        MeshPart part = new MeshPart(mesh, null);

        // An error is thrown as `UnexpectedResultException`.
        std_vector_MR_FaceFace selfCollidingPairs = Functions.findSelfCollidingTriangles(part, null, null, null); // find self-intersecting faces pairs
        for (long i = 0; i < selfCollidingPairs.size(); i++)
        {
            FaceFace ff = selfCollidingPairs.at(i);
            System.out.println(ff.aFace().id_() + " " + ff.bFace().id_()); // print each pair
        }

        FaceBitSet selfCollidingBitSet = Functions.findSelfCollidingTrianglesBS(part, null, null, null); // find bitset of self-intersecting faces
        System.out.println(selfCollidingBitSet.count()); // print number of self-intersecting faces

        boolean isSelfColliding = Functions.findSelfCollidingTriangles(part, null, null, null, null); // fast check if mesh has self-intersections
        System.out.println(isSelfColliding);
    }
}
