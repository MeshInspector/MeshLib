import meshlib.FixMeshDegeneraciesParams;
import meshlib.Functions;
import meshlib.Mesh;

// The class isn't public because of the `.dox.java` file name, the launcher doesn't mind.
class MeshFixDegeneracies
{
    public static void main(String[] args)
    {
        // An error is thrown as `UnexpectedResultException`.
        Mesh mesh = meshlib.meshload.Functions.fromAnySupportedFormat("mesh.stl", null);

        // you can set various parameters for the resolving process; see the documentation for more info
        FixMeshDegeneraciesParams params = new FixMeshDegeneraciesParams();
        params.setMaxDeviation(1e-5f * mesh.computeBoundingBox(null).diagonal());
        params.setTinyEdgeLength(1e-3f);
        Functions.fixMeshDegeneracies(mesh, params);

        // Save result
        meshlib.meshsave.Functions.toAnySupportedFormat(mesh, "fixedMesh.stl", null);
    }
}
