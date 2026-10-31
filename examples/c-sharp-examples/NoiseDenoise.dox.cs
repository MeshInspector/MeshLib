public static class NoiseDenoiseExample
{
    public static void Main(string[] args)
    {
        try
        {
            // Load mesh
            var mesh = MR.MeshLoad.fromAnySupportedFormat("mesh.stl");

            // Add noise to the mesh
            MR.NoiseSettings noiseSettings = new();
            noiseSettings.sigma = mesh.computeBoundingBox().diagonal() * 0.0001f;
            MR.addNoise(mesh.points, mesh.topology.getValidVerts(), noiseSettings);

            // Invalidate the mesh because of the external vertex changes
            mesh.invalidateCaches();

            // Save the noised mesh
            MR.MeshSave.toAnySupportedFormat(mesh, "noised_mesh.stl");

            // Denoise the mesh with sharpening for sharp edges
            // see the article "Mesh Denoising via a Novel Mumford-Shah Framework"
            MR.meshDenoiseViaNormals(mesh);

            // Save the denoised mesh
            MR.MeshSave.toAnySupportedFormat(mesh, "denoised_mesh.stl");
        }
        catch (Exception e)
        {
            Console.WriteLine("Error: {0}", e.Message);
        }
    }
}
