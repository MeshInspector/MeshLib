public class MeshICPExample
{
    public static void Run()
    {
        try
        {
            // Load meshes

            var mesh_floating = MR.MeshLoad.fromAnySupportedFormat("meshA.stl");
            var mesh_fixed = MR.MeshLoad.fromAnySupportedFormat("meshB.stl");

            MR.MeshOrPointsXf mesh_xf_floating = new(mesh_floating, new MR.AffineXf3f());
            MR.MeshOrPointsXf mesh_xf_fixed = new(mesh_fixed, new MR.AffineXf3f());

            // Prepare ICP parameters
            float diagonal = mesh_xf_fixed.obj.computeBoundingBox().diagonal();
            float icpSamplingVoxelSize = diagonal * 0.01f; // To sample points from object
            MR.ICPProperties icpParams = new();
            icpParams.distThresholdSq = diagonal * diagonal * 0.01f; // Use points pairs with maximum distance specified
            icpParams.exitVal = diagonal * 0.003f; // Stop when distance reached

            // Calculate transformation
            MR.ICP icp = new(mesh_xf_floating, mesh_xf_fixed, icpSamplingVoxelSize);
            icp.setParams(icpParams);
            MR.AffineXf3f xf = icp.calculateTransformation();

            // Transform floating mesh
            mesh_floating.transform(xf);

            // Output information string
            Console.WriteLine("info {0}", icp.getStatusInfo());

            // Save result
            MR.MeshSave.toAnySupportedFormat(mesh_floating, "meshA_icp.stl");
        }
        catch (Exception e)
        {
            Console.WriteLine("Error: {0}", e.Message);
        }
    }
}
