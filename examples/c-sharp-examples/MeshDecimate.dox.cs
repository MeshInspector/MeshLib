public class MeshDecimateExample
{
    public static void Run(string[] args)
    {
        try
        {
            // Load mesh
            MR.Expected_MRMesh_StdString mesh_ex = MR.MeshLoad.FromAnySupportedFormat("mesh.stl");
            if (mesh_ex.GetError() is var mesh_error and not null)
                throw new Exception(mesh_error);

            MR.Mesh mesh = mesh_ex.GetValue()!;

            // Setup decimate parameters
            MR.DecimateSettings ds = new();
            ds.Strategy = MR.DecimateStrategy.MinimizeError;
            ds.MaxError = 1e-5f * mesh.ComputeBoundingBox().Diagonal();
            ds.TinyEdgeLength = 1e-3f;
            ds.PackMesh = true;

            // Decimate mesh
            MR.DecimateResult result = MR.DecimateMesh(mesh, ds);

            // Save result
            MR.Expected_Void_StdString save_ex = MR.MeshSave.ToAnySupportedFormat(mesh, "decimated_mesh.stl");
            if (save_ex.GetError() is var save_error and not null)
                throw new Exception(save_error);
        }
        catch (Exception e)
        {
            Console.WriteLine("Error: {0}", e.Message);
        }
    }
}
