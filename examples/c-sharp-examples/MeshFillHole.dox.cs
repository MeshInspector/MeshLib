using System.Reflection;
using static MR.DotNet;

public static class MeshFillHoleExample
{
    public static void Run(string[] args)
    {
        try
        {
            if (args.Length != 2 && args.Length != 3)
            {
                Console.WriteLine("Usage: {0} MeshFillHoleExample INPUT [OUTPUT]", Assembly.GetExecutingAssembly().GetName().Name);
                return;
            }

            string inputFile = args[1];
            string outputFile = args.Length == 3 ? args[2] : inputFile;

            var mesh = MeshLoad.FromAnySupportedFormat(inputFile);
            var holes = mesh.HoleRepresentiveEdges;

            var fillHoleParams = new FillHoleParams();
            fillHoleParams.Metric = FillHoleMetric.GetUniversalMetric( mesh );
            fillHoleParams.OutNewFaces = new FaceBitSet();
            
            FillHoles(ref mesh, holes.ToList(), fillHoleParams);
            Console.WriteLine("Number of new faces: {0}", fillHoleParams.OutNewFaces.Count());

            MeshSave.ToAnySupportedFormat(mesh, outputFile);
        }
        catch (Exception e)
        {
            Console.WriteLine("Error: {0}", e.Message);
        }
    }
}
