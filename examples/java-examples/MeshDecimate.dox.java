import meshlib.mr.DecimateResult;
import meshlib.mr.DecimateSettings;
import meshlib.mr.Functions;
import meshlib.mr.Mesh;
import meshlib.mr.expected_MR_Mesh_std_string;
import meshlib.mr.expected_void_std_string;

// The class isn't public because of the `.dox.java` file name, the launcher doesn't mind.
class MeshDecimate
{
    public static void main(String[] args)
    {
        // Load mesh.
        expected_MR_Mesh_std_string loaded = meshlib.mr.meshload.Functions.fromAnySupportedFormat("mesh.stl", null);
        if (!loaded.success())
        {
            System.err.println("Failed to load mesh: " + loaded.error());
            System.exit(1);
        }
        Mesh mesh = loaded.value_mut();

        // Repack mesh optimally.
        // It's not necessary but highly recommended to achieve the best performance in parallel processing.
        mesh.packOptimally(null);

        // Setup decimate parameters.
        DecimateSettings settings = new DecimateSettings();

        // Decimation stop thresholds, you may specify one or both.
        settings.setMaxDeletedFaces(1000); // Number of faces to be deleted.
        settings.setMaxError(0.05f); // Maximum error when decimation stops.

        // Number of parts for parallel processing, greatly improves performance at the cost of minor quality loss.
        // A value close to the number of CPU cores is usually the fastest; much larger values add per-part overhead.
        settings.setSubdivideParts(16);

        // Decimate mesh.
        DecimateResult result = Functions.decimateMesh(mesh, settings);
        System.out.println("Removed " + result.vertsDeleted() + " vertices, " + result.facesDeleted() + " faces");

        // Save result.
        expected_void_std_string saved = meshlib.mr.meshsave.Functions.toAnySupportedFormat(mesh, "decimated_mesh.stl", null);
        if (!saved.success())
        {
            System.err.println("Failed to save mesh: " + saved.error());
            System.exit(1);
        }
    }
}
