import meshlib.Mesh;
import meshlib.UnexpectedResultException;

// The class isn't public because of the `.dox.java` file name, the launcher doesn't mind.
class MeshLoadSave
{
    public static void main(String[] args)
    {
        String inputPath = args.length > 0 ? args[0] : "mesh.stl";
        String outputPath = args.length > 1 ? args[1] : "mesh.ply";

        // Load mesh. The C++ function returns `Expected<Mesh>`: the Java wrapper returns the mesh, or throws the error.
        Mesh mesh;
        try
        {
            mesh = meshlib.meshload.Functions.fromAnySupportedFormat(inputPath, null);
        }
        catch (UnexpectedResultException e)
        {
            if (args.length == 0)
                System.err.println("Usage: MeshLoadSave [input mesh] [output mesh]");
            System.err.println("Failed to load mesh: " + e.getMessage());
            System.exit(1);
            return;
        }

        // Save mesh. An error is thrown the same way.
        meshlib.meshsave.Functions.toAnySupportedFormat(mesh, outputPath, null);
    }
}
