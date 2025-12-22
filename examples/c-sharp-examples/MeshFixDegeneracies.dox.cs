using System.Reflection;

public static class MeshFixDegeneraciesExample
{
    public static void Run(string[] args)
    {
        try
        {
            if (args.Length != 2 && args.Length != 3)
            {
                Console.WriteLine("Usage: {0} MeshFixDegeneraciesExample INPUT [OUTPUT]", Assembly.GetExecutingAssembly().GetName().Name);
                return;
            }

            string inputFile = args[1];
            string outputFile = args.Length == 3 ? args[2] : inputFile;

            MR.Expected_MRMesh_StdString mesh_ex = MR.MeshLoad.FromAnySupportedFormat(inputFile);
            if (mesh_ex.GetError() is var mesh_error and not null)
                throw new Exception(mesh_error);
            MR.Mesh mesh = mesh_ex.GetValue()!;

            MR.FixMeshDegeneraciesParams parameters = new();
            parameters.MaxDeviation = mesh.ComputeBoundingBox().Diagonal() * 1e-5f;
            parameters.TinyEdgeLength = 1e-3f;

            MR.FixMeshDegeneracies(mesh, parameters);
            MR.Expected_Void_StdString save_ex = MR.MeshSave.ToAnySupportedFormat(mesh, outputFile);
            if (save_ex.GetError() is var save_error and not null)
                throw new Exception(save_error);
        }
        catch (Exception e)
        {
            Console.WriteLine("Error: {0}", e.Message);
        }
    }
}
