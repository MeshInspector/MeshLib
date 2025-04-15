using System.Reflection;
using static MR.DotNet;

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

            var mesh = MeshLoad.FromAnySupportedFormat(inputFile);
            var parameters = new FixMeshDegeneraciesParams();
            parameters.maxDeviation = mesh.BoundingBox.Diagonal() * 1e-3f;
            parameters.tinyEdgeLength = parameters.maxDeviation * 0.1f;

            FixMeshDegeneracies(ref mesh, parameters);
        }
        catch (Exception e)
        {
            Console.WriteLine("Error: {0}", e.Message);
        }
    }
}
