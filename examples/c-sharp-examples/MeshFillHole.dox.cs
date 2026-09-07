using System.Reflection;

public static class MeshFillHoleExample
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

            // INPUT is a mesh file you supply; this example needs a mesh that has holes to fill
            string inputFile = args[0];
            string outputFile = args.Length == 2 ? args[1] : inputFile;

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
