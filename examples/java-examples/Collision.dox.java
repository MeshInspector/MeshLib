import meshlib.AffineXf3f;
import meshlib.FaceFace;
import meshlib.Functions;
import meshlib.Mesh;
import meshlib.MeshPart;
import meshlib.Vector3f;
import meshlib.std_pair_MR_FaceBitSet_MR_FaceBitSet;
import meshlib.std_vector_MR_FaceFace;

// The class isn't public because of the `.dox.java` file name, the launcher doesn't mind.
class Collision
{
    public static void main(String[] args)
    {
        Mesh meshA = Functions.makeUVSphere(null, null, null); // make mesh A
        Mesh meshB = Functions.makeUVSphere(null, null, null); // make mesh B
        meshB.transform(AffineXf3f.translation(new Vector3f(0.1f, 0.1f, 0.1f)), null); // shift mesh B for better demonstration

        MeshPart partA = new MeshPart(meshA, null);
        MeshPart partB = new MeshPart(meshB, null);

        std_vector_MR_FaceFace collidingFacePairs = Functions.findCollidingTriangles(partA, partB, null, null); // find each pair of colliding faces
        for (long i = 0; i < collidingFacePairs.size(); i++)
        {
            FaceFace ff = collidingFacePairs.at(i);
            System.out.println(ff.aFace().id_() + " " + ff.bFace().id_()); // print each pair of colliding faces
        }

        std_pair_MR_FaceBitSet_MR_FaceBitSet collidingFaceBitSets = Functions.findCollidingTriangleBitsets(partA, partB, null); // find bitsets of colliding faces
        System.out.println(collidingFaceBitSets.first().count()); // print number of colliding faces from mesh A
        System.out.println(collidingFaceBitSets.second().count()); // print number of colliding faces from mesh B

        boolean isColliding = !Functions.findCollidingTriangles(partA, partB, null, true).empty(); // fast check if mesh A and mesh B collide
        System.out.println(isColliding);
    }
}
