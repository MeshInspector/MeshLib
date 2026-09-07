using System.Reflection;

public static class MeshFixDegeneraciesExample
{
    public static void Main(string[] args)
    {
        try
        {
            if (args.Length != 1 && args.Length != 2)
            {
                Console.WriteLine("Usage: {0} INPUT [OUTPUT]", Assembly.GetExecutingAssembly().GetName().Name);
                return;
            }

            // INPUT is a mesh file you supply; this example needs a mesh that has degeneracies to fix
            string inputFile = args[0];
            string outputFile = args.Length == 2 ? args[1] : inputFile;

            var mesh = MR.MeshLoad.fromAnySupportedFormat(inputFile);

            MR.FixMeshDegeneraciesParams parameters = new();
            parameters.maxDeviation = mesh.computeBoundingBox().diagonal() * 1e-5f;
            parameters.tinyEdgeLength = 1e-3f;

            MR.fixMeshDegeneracies(mesh, parameters);
            MR.MeshSave.toAnySupportedFormat(mesh, outputFile);
        }
        catch (Exception e)
        {
            Console.WriteLine("Error: {0}", e.Message);
        }
    }
}
