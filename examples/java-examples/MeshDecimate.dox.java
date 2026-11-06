import meshlib.DecimateSettings;
import meshlib.Functions;
import meshlib.Mesh;

// The class isn't public because of the `.dox.java` file name, the launcher doesn't mind.
class MeshDecimate
{
    public static void main(String[] args)
    {
        // Load mesh. An error is thrown as `UnexpectedResultException`.
        Mesh mesh = meshlib.meshload.Functions.fromAnySupportedFormat("mesh.stl", null);

        // Repack mesh optimally without building an AABB tree, which decimation invalidates anyway.
        // It's not necessary but highly recommended to achieve the best performance in parallel processing.
        mesh.packOptimally(false);

        // Setup decimate parameters.
        DecimateSettings settings = new DecimateSettings();

        // Decimation stop thresholds, you may specify one or both.
        settings.setMaxDeletedFaces(1000); // Number of faces to be deleted.
        settings.setMaxError(0.05f); // Maximum error when decimation stops.

        // Number of parts for parallel processing, greatly improves performance at the cost of minor quality loss.
        // 16 is a good choice even with fewer CPU cores, since smaller parts are processed faster; much larger values add per-part overhead.
        settings.setSubdivideParts(16);

        // Decimate mesh.
        Functions.decimateMesh(mesh, settings);

        // Save result.
        meshlib.meshsave.Functions.toAnySupportedFormat(mesh, "decimated_mesh.stl", null);
    }
}
