import meshlib.Box3f;
import meshlib.FreeFormDeformer;
import meshlib.Mesh;
import meshlib.Vector3i;

// The class isn't public because of the `.dox.java` file name, the launcher doesn't mind.
class FreeFormDeformation
{
    public static void main(String[] args)
    {
        // Load mesh. An error is thrown as `UnexpectedResultException`.
        Mesh mesh = meshlib.meshload.Functions.fromAnySupportedFormat("mesh.stl", null);

        // Construct deformer on mesh vertices
        FreeFormDeformer ffDeformer = new FreeFormDeformer(mesh.points(), mesh.topology().getValidVerts());

        // Compute mesh bounding box
        Box3f box = mesh.computeBoundingBox(null);

        // Init deformer with 3x3 grid on mesh box
        ffDeformer.init(Vector3i.diagonal(3), box);

        // Move some control points of the grid to the center
        ffDeformer.setRefGridPointPosition(new Vector3i(1, 1, 0), box.center());
        ffDeformer.setRefGridPointPosition(new Vector3i(1, 1, 2), box.center());
        ffDeformer.setRefGridPointPosition(new Vector3i(0, 1, 1), box.center());
        ffDeformer.setRefGridPointPosition(new Vector3i(2, 1, 1), box.center());
        ffDeformer.setRefGridPointPosition(new Vector3i(1, 0, 1), box.center());
        ffDeformer.setRefGridPointPosition(new Vector3i(1, 2, 1), box.center());

        // Apply the deformation to the mesh vertices
        ffDeformer.apply();

        // Invalidate the mesh because of external vertex changes
        mesh.invalidateCaches(null);

        // Save deformed mesh
        meshlib.meshsave.Functions.toAnySupportedFormat(mesh, "deformed_mesh.stl", null);
    }
}
