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

            MR.Expected_MRMesh_StdString mesh_ex = MR.MeshLoad.FromAnySupportedFormat(inputFile);
            if (mesh_ex.GetError() is var mesh_error and not null)
                throw new Exception(mesh_error);
            MR.Mesh mesh = mesh_ex.GetValue()!;

            MR.Std.Vector_MREdgeId holes = mesh.Topology.FindHoleRepresentiveEdges();

            MR.FillHoleParams fillHoleParams = new();
            fillHoleParams.Metric.Assign(MR.GetUniversalMetric(mesh));
            MR.FaceBitSet outfaces = new();
            // TODO
            // fillHoleParams.OutNewFaces = ...

            MR.FillHoles(mesh, holes, fillHoleParams);
            // TODO
            // Console.WriteLine("Number of new faces: {0}", fillHoleParams.OutNewFaces.Count());

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
