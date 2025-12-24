using System.Reflection;

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

            var mesh = MR.MeshLoad.FromAnySupportedFormat(inputFile);

            MR.Std.Vector_MREdgeId holes = mesh.Topology.FindHoleRepresentiveEdges();

            MR.FillHoleParams fillHoleParams = new();
            fillHoleParams.Metric.Assign(MR.GetUniversalMetric(mesh));
            MR.FaceBitSet outfaces = new();
            // TODO
            // fillHoleParams.OutNewFaces = ...

            MR.FillHoles(mesh, holes, fillHoleParams);
            // TODO
            // Console.WriteLine("Number of new faces: {0}", fillHoleParams.OutNewFaces.Count());

            MR.MeshSave.ToAnySupportedFormat(mesh, outputFile);
        }
        catch (Exception e)
        {
            Console.WriteLine("Error: {0}", e.Message);
        }
    }
}
