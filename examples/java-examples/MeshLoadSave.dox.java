import meshlib.mr.Mesh;
import meshlib.mr.expected_MR_Mesh_std_string;
import meshlib.mr.expected_void_std_string;

// The class isn't public because of the `.dox.java` file name, the launcher doesn't mind.
class MeshLoadSave
{
    public static void main(String[] args)
    {
        String inputPath = args.length > 0 ? args[0] : "mesh.stl";
        String outputPath = args.length > 1 ? args[1] : "mesh.ply";

        // Load mesh.
        expected_MR_Mesh_std_string loaded = meshlib.mr.meshload.Functions.fromAnySupportedFormat(inputPath, null);
        if (!loaded.success())
        {
            if (args.length == 0)
                System.err.println("Usage: MeshLoadSave [input mesh] [output mesh]");
            System.err.println("Failed to load mesh: " + loaded.error());
            System.exit(1);
        }
        Mesh mesh = loaded.value();

        // Save mesh.
        expected_void_std_string saved = meshlib.mr.meshsave.Functions.toAnySupportedFormat(mesh, outputPath, null);
        if (!saved.success())
        {
            System.err.println("Failed to save mesh: " + saved.error());
            System.exit(1);
        }
    }
}
