import meshlib.Mesh;
import meshlib.SaveSettings;
import meshlib.UnexpectedResultException;

// The class isn't public because of the `.dox.java` file name, the launcher doesn't mind.
class MeshLoadSave
{
    public static void main(String[] args)
    {
        // Load mesh. The C++ function returns `Expected<Mesh>`: the Java wrapper returns the mesh, or throws the error.
        Mesh mesh;
        try
        {
            mesh = meshlib.meshload.Functions.fromAnySupportedFormat("mesh.stl", null);
        }
        catch (UnexpectedResultException e)
        {
            System.err.println("Failed to load mesh: " + e.getMessage());
            System.exit(1);
            return;
        }

        // Save mesh. An error is thrown the same way.
        meshlib.meshsave.Functions.toAnySupportedFormat(mesh, "mesh.ply", null);

        // More mesh formats are available in the MRIOExtras library
        meshlib.meshsave.Functions.toCtm(mesh, "mesh.ctm", new SaveSettings());
    }
}
