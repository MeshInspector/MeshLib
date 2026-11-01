import meshlib.Functions;
import meshlib.Mesh;
import meshlib.MeshMeshSignedDistanceResult;
import meshlib.MeshPart;

// The class isn't public because of the `.dox.java` file name, the launcher doesn't mind.
class SignedDistance
{
    public static void main(String[] args)
    {
        // An error is thrown as `UnexpectedResultException`.
        Mesh mesh1 = meshlib.meshload.Functions.fromAnySupportedFormat("mesh1.ctm", null);
        Mesh mesh2 = meshlib.meshload.Functions.fromAnySupportedFormat("mesh2.ctm", null);

        MeshMeshSignedDistanceResult dist = Functions.findSignedDistance(new MeshPart(mesh1, null), new MeshPart(mesh2, null), null, null);
        System.out.println("Signed distance between meshes is " + dist.signedDist());
    }
}
