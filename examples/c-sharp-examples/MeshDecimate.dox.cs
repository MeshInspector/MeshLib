public class MeshDecimateExample
{
    public static void Run(string[] args)
    {
        try
        {
            // Load mesh
            var mesh = MR.MeshLoad.FromAnySupportedFormat("mesh.stl");

            // Setup decimate parameters
            MR.DecimateSettings ds = new();
            ds.Strategy = MR.DecimateStrategy.MinimizeError;
            ds.MaxError = 1e-5f * mesh.ComputeBoundingBox().Diagonal();
            ds.TinyEdgeLength = 1e-3f;
            ds.PackMesh = true;

            // Decimate mesh
            MR.DecimateResult result = MR.DecimateMesh(mesh, ds);

            // Save result
            MR.MeshSave.ToAnySupportedFormat(mesh, "decimated_mesh.stl");
        }
        catch (Exception e)
        {
            Console.WriteLine("Error: {0}", e.Message);
        }
    }
}
