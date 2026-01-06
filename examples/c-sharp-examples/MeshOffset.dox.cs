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
            var mesh = MR.MeshLoad.FromAnySupportedFormat("mesh.stl");

            MR.MeshPart mp = new(mesh);

            // Setup parameters
            MR.OffsetParameters op = new();
            op.voxelSize = MR.SuggestVoxelSize(mp, 1e6f);

            // Make offset mesh
            var result = MR.OffsetMesh(mp, offsetValue, op);

            // Save result
            MR.MeshSave.ToAnySupportedFormat(result, "mesh_offset.stl");
        }
        catch (Exception e)
        {
            Console.WriteLine("Error: {0}", e.Message);
        }
    }
}
