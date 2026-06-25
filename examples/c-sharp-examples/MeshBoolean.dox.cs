using System.Reflection;

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
            var mesh_a = MR.MeshLoad.fromAnySupportedFormat(args[1]);
            var mesh_b = MR.MeshLoad.fromAnySupportedFormat(args[2]);

            // perform boolean operation
            MR.BooleanResult res = MR.boolean(mesh_a, mesh_b, MR.BooleanOperation.Intersection);

            // save result to STL file
            MR.MeshSave.toAnySupportedFormat(res.mesh, "out_boolean.stl");
        }
        catch (Exception e)
        {
            Console.WriteLine("Error: {0}", e.Message);
        }
    }
}
