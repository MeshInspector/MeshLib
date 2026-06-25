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

            var mesh = MR.MeshLoad.fromAnySupportedFormat(inputFile);

            MR.Std.Vector_MREdgeId holes = mesh.topology.findHoleRepresentiveEdges();

            MR.FillHoleParams fillHoleParams = new();
            fillHoleParams.metric.assign(MR.getUniversalMetric(mesh));
            MR.FaceBitSet outfaces = new();
            // TODO
            // fillHoleParams.OutNewFaces = ...

            MR.fillHoles(mesh, holes, fillHoleParams);
            // TODO
            // Console.WriteLine("Number of new faces: {0}", fillHoleParams.OutNewFaces.Count());

            MR.MeshSave.toAnySupportedFormat(mesh, outputFile);
        }
        catch (Exception e)
        {
            Console.WriteLine("Error: {0}", e.Message);
        }
    }
}
