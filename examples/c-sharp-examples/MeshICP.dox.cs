public class MeshICPExample
{
    public static void Run()
    {
        try
        {
            // Load meshes

            MR.Expected_MRMesh_StdString mesh_floating_ex = MR.MeshLoad.FromAnySupportedFormat("meshA.stl");
            if (mesh_floating_ex.GetError() is var mesh_floating_error and not null)
                throw new Exception(mesh_floating_error);
            MR.Mesh mesh_floating = mesh_floating_ex.GetValue()!;

            MR.Expected_MRMesh_StdString mesh_fixed_ex = MR.MeshLoad.FromAnySupportedFormat("meshB.stl");
            if (mesh_fixed_ex.GetError() is var mesh_fixed_error and not null)
                throw new Exception(mesh_fixed_error);
            MR.Mesh mesh_fixed = mesh_fixed_ex.GetValue()!;

            MR.MeshOrPointsXf mesh_xf_floating = new(mesh_floating, new MR.AffineXf3f());
            MR.MeshOrPointsXf mesh_xf_fixed = new(mesh_fixed, new MR.AffineXf3f());

            // Prepare ICP parameters
            float diagonal = mesh_xf_fixed.Obj.ComputeBoundingBox().Diagonal();
            float icpSamplingVoxelSize = diagonal * 0.01f; // To sample points from object
            MR.ICPProperties icpParams = new();
            icpParams.DistThresholdSq = diagonal * diagonal * 0.01f; // Use points pairs with maximum distance specified
            icpParams.ExitVal = diagonal * 0.003f; // Stop when distance reached

            // Calculate transformation
            MR.ICP icp = new(mesh_xf_floating, mesh_xf_fixed, icpSamplingVoxelSize);
            icp.SetParams(icpParams);
            MR.AffineXf3f xf = icp.CalculateTransformation();

            // Transform floating mesh
            mesh_floating.Transform(xf);

            // Output information string
            Console.WriteLine("info {0}", icp.GetStatusInfo());

            // Save result
            MR.Expected_Void_StdString save_ex = MR.MeshSave.ToAnySupportedFormat(mesh_floating, "meshA_icp.stl");
            if (save_ex.GetError() is var save_error and not null)
                throw new Exception(save_error);
        }
        catch (Exception e)
        {
            Console.WriteLine("Error: {0}", e.Message);
        }
    }
}
