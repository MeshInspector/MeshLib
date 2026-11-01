public static class MeshDecimateExample
{
    public static void Main(string[] args)
    {
        // Load mesh
        var mesh = MR.MeshLoad.fromAnySupportedFormat("mesh.stl");

        // Repack mesh optimally without building an AABB tree, which decimation invalidates anyway.
        // It's not necessary but highly recommended to achieve the best performance in parallel processing
        mesh.packOptimally(false);

        // Setup decimate parameters
        MR.DecimateSettings settings = new();

        // Decimation stop thresholds, you may specify one or both
        settings.maxDeletedFaces = 1000; // Number of faces to be deleted
        settings.maxError = 0.05f; // Maximum error when decimation stops

        // Number of parts for parallel processing, greatly improves performance at the cost of minor quality loss.
        // 16 is a good choice even with fewer CPU cores, since smaller parts are processed faster; much larger values add per-part overhead
        settings.subdivideParts = 16;

        // Decimate mesh
        MR.decimateMesh(mesh, settings);

        // Save result
        MR.MeshSave.toAnySupportedFormat(mesh, "decimated_mesh.stl");
    }
}
