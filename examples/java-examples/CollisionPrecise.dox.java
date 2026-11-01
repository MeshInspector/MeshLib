import meshlib.AffineXf3f;
import meshlib.CoordinateConverters;
import meshlib.Functions;
import meshlib.Mesh;
import meshlib.MeshPart;
import meshlib.VarEdgeTri;
import meshlib.Vector3f;
import meshlib.Vector_MR_VarEdgeTri;

// The class isn't public because of the `.dox.java` file name, the launcher doesn't mind.
class CollisionPrecise
{
    public static void main(String[] args)
    {
        Mesh meshA = Functions.makeUVSphere(null, null, null); // make mesh A
        Mesh meshB = Functions.makeUVSphere(null, null, null); // make mesh B
        meshB.transform(AffineXf3f.translation(new Vector3f(0.1f, 0.1f, 0.1f)), null); // shift mesh B for better demonstration

        MeshPart partA = new MeshPart(meshA, null);
        MeshPart partB = new MeshPart(meshB, null);

        CoordinateConverters converters = Functions.getVectorConverters(partA, partB, null); // create converters to integer field (needed for absolute precision predicates)
        Vector_MR_VarEdgeTri collidingFaceEdges = Functions.findCollidingEdgeTrisPrecise(partA, partB, converters.toInt(), null, null); // find each intersecting edge/triangle pair
        // print pairs of edges triangles
        for (long i = 0; i < collidingFaceEdges.size(); i++)
        {
            VarEdgeTri vet = collidingFaceEdges.index(i);
            if (vet.isEdgeATriB())
                System.out.println("edgeA: " + vet.edge().id_() + ", triB: " + vet.tri().id_());
            else
                System.out.println("triA: " + vet.tri().id_() + ", edgeB: " + vet.edge().id_());
        }
    }
}
