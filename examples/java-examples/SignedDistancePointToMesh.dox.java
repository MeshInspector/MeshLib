import meshlib.Functions;
import meshlib.Mesh;
import meshlib.MeshPart;
import meshlib.Vector3f;
import meshlib.std_optional_MR_SignedDistanceToMeshResult;

// The class isn't public because of the `.dox.java` file name, the launcher doesn't mind.
class SignedDistancePointToMesh
{
    public static void main(String[] args)
    {
        // An error is thrown as `UnexpectedResultException`.
        Mesh mesh1 = meshlib.meshload.Functions.fromAnySupportedFormat("mesh1.ctm", null);
        Vector3f point = new Vector3f(1.5f, 1.0f, 0.5f);

        std_optional_MR_SignedDistanceToMeshResult dist = Functions.findSignedDistance(point, new MeshPart(mesh1, null), null, null);
        if (dist.has_value())
            System.out.println("Signed distance from point to mesh " + dist.value().dist());
    }
}
