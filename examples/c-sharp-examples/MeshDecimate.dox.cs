using static MR.DotNet;

public class MeshDecimateExample
{
    public static void Run(string[] args)
    {
        try
        {
            // Load mesh
            var mesh = MeshLoad.FromAnySupportedFormat("mesh.stl");

            // Setup decimate parameters
            DecimateParameters dp = new DecimateParameters();
            dp.strategy = DecimateStrategy.MinimizeError;
            dp.maxError = 1e-5f * mesh.BoundingBox.Diagonal();
            dp.tinyEdgeLength = 1e-3f;
            dp.packMesh = true;

            // Decimate mesh
            var result = Decimate(ref mesh, dp);

            // Save result
            MeshSave.ToAnySupportedFormat(mesh, "decimated_mesh.stl");
        }
        catch (Exception e)
        {
            Console.WriteLine("Error: {0}", e.Message);
        }
    }
}