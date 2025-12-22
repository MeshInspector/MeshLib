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
            MR.Expected_MRMesh_StdString mesh_a_ex = MR.MeshLoad.FromAnySupportedFormat(args[1]);
            if (mesh_a_ex.GetError() is var error_a and not null)
                throw new Exception(error_a);
            MR.Expected_MRMesh_StdString mesh_b_ex = MR.MeshLoad.FromAnySupportedFormat(args[2]);
            if (mesh_b_ex.GetError() is var error_b and not null)
                throw new Exception(error_b);

            // perform boolean operation
            MR.BooleanResult res = MR.Boolean(mesh_a_ex.GetValue()!, mesh_b_ex.GetValue()!, MR.BooleanOperation.Intersection);

            // save result to STL file
            MR.MeshSave.ToAnySupportedFormat(res.Mesh, "out_boolean.stl");
        }
        catch (Exception e)
        {
            Console.WriteLine("Error: {0}", e.Message);
        }
    }
}
