using System.Reflection;
using static MR.DotNet;

public static class MeshResolveDegenerationsExample
{
    public static void Run(string[] args)
    {
        try
        {
            if (args.Length != 2 && args.Length != 3)
            {
                Console.WriteLine("Usage: {0} MeshResolveDegenerationsExample INPUT [OUTPUT]", Assembly.GetExecutingAssembly().GetName().Name);
                return;
            }

            string inputFile = args[1];
            string outputFile = args.Length == 3 ? args[2] : inputFile;

            var mesh = MeshLoad.FromAnySupportedFormat(inputFile);
            var parameters = new ResolveMeshDegenParameters();
            parameters.maxDeviation = mesh.BoundingBox.Diagonal() * 1e-5f;
            parameters.tinyEdgeLength = 1e-3f;

            if (!ResolveMeshDegenerations(ref mesh, parameters))
            {
                Console.WriteLine("No changes were made");
            }
            else
            {
                Console.WriteLine("Degenerations resolved");
                MeshSave.ToAnySupportedFormat(mesh, outputFile);
            }
        }
        catch (Exception e)
        {
            Console.WriteLine("Error: {0}", e.Message);
        }
    }
}
