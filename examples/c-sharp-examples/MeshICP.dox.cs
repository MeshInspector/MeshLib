public class MeshICPExample
{
    public static void Main(string[] args)
    {
        try
        {
            // Make two meshes to align
            // (to align your own meshes, load them with MR.MeshLoad.fromAnySupportedFormat instead)
            var mesh_floating = MR.makeTorus(2.0f, 1.0f, 32, 32);
            var mesh_fixed = MR.makeTorus(2.0f, 1.0f, 32, 32);

            // Displace the floating mesh, so ICP has a transformation to recover
            mesh_floating.transform(new MR.AffineXf3f(
                MR.Matrix3f.rotation(new MR.Vector3f(1.0f, 0.0f, 0.0f), 0.2f),
                new MR.Vector3f(0.3f, 0.2f, 0.1f)));

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
