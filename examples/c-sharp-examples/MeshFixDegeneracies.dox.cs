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

            var mesh = MR.MeshLoad.FromAnySupportedFormat(inputFile);

            MR.FixMeshDegeneraciesParams parameters = new();
            parameters.MaxDeviation = mesh.ComputeBoundingBox().Diagonal() * 1e-5f;
            parameters.TinyEdgeLength = 1e-3f;

            MR.FixMeshDegeneracies(mesh, parameters);
            MR.MeshSave.ToAnySupportedFormat(mesh, outputFile);
        }
        catch (Exception e)
        {
            Console.WriteLine("Error: {0}", e.Message);
        }
    }
}
