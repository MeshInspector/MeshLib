using System.Globalization;
using System.Reflection;

public class MeshOffsetExample
{
    public static void Run(string[] args)
    {
        if (args.Length != 2)
        {
            Console.WriteLine("Usage: {0} MeshOffsetExample OFFSET_VALUE", Assembly.GetExecutingAssembly().GetName().Name);
            return;
        }

        try
        {
            float offsetValue = float.Parse(args[1],
                      System.Globalization.NumberStyles.AllowDecimalPoint,
                      CultureInfo.InvariantCulture);

            // Load mesh
            MR.Expected_MRMesh_StdString mesh_ex = MR.MeshLoad.FromAnySupportedFormat("mesh.stl");
            if (mesh_ex.GetError() is var mesh_error and not null)
                throw new Exception(mesh_error);

            MR.MeshPart mp = new(mesh_ex.GetValue()!);

            // Setup parameters
            MR.OffsetParameters op = new();
            op.VoxelSize = MR.SuggestVoxelSize(mp, 1e6f);

            // Make offset mesh
            MR.Expected_MRMesh_StdString result_ex = MR.OffsetMesh(mp, offsetValue, op);
            if (result_ex.GetError() is var result_error and not null)
                throw new Exception(result_error);

            // Save result
            MR.Expected_Void_StdString save_ex = MR.MeshSave.ToAnySupportedFormat(result_ex.GetValue()!, "mesh_offset.stl");
            if (save_ex.GetError() is var save_error and not null)
                throw new Exception(save_error);
        }
        catch (Exception e)
        {
            Console.WriteLine("Error: {0}", e.Message);
        }
    }
}
