public class MeshDecimateExample
{
    public static void Main(string[] args)
    {
        try
        {
            // Make a mesh to decimate
            // (to decimate your own mesh, load it with MR.MeshLoad.fromAnySupportedFormat("mesh.stl") instead)
            var mesh = MR.makeUVSphere(1.0f, 64, 64);

            // Repack mesh optimally.
            // It's not necessary but highly recommended to achieve the best performance in parallel processing
            mesh.packOptimally();

            // Setup decimate parameters
            MR.DecimateSettings ds = new();

            // Decimation stop thresholds, you may specify one or both
            ds.maxDeletedFaces = 1000; // Number of faces to be deleted
            ds.maxError = 0.05f; // Maximum error when decimation stops

            // Number of parts to simultaneous processing, greatly improves performance by cost of minor quality loss.
            // Recommended to set to the number of available CPU cores or more for the best performance
            ds.subdivideParts = 64;

            // Decimate mesh
            MR.decimateMesh(mesh, ds);

            // Save result
            MR.MeshSave.toAnySupportedFormat(mesh, "decimated_mesh_cs.stl");
        }
        catch (Exception e)
        {
            Console.WriteLine("Error: {0}", e.Message);
        }
    }
}
