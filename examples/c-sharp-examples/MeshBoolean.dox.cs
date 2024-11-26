using System.Reflection;
using static MR.DotNet;

public class MeshBooleanExample
{
    public static void Run(string[] args)
    {
        if (args.Length != 3)
        {
            Console.WriteLine("Usage: {0} MeshBooleanExample INPUT1 INPUT2", Assembly.GetExecutingAssembly().GetName().Name);
            return;
        }

        try
        {
            // load mesh
            Mesh meshA = MeshLoad.FromAnySupportedFormat(args[1]);
            Mesh meshB = MeshLoad.FromAnySupportedFormat(args[2]);

            // perform boolean operation
            var res = Boolean(meshA, meshB, BooleanOperation.Intersection);

            // save result to STL file
            MeshSave.ToAnySupportedFormat(res.mesh, "out_boolean.stl");
        }
        catch (Exception e)
        {
            Console.WriteLine("Error: {0}", e.Message);
        }
    }
}