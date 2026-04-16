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
            var mesh = MR.MeshLoad.fromAnySupportedFormat("mesh.stl");

            MR.MeshPart mp = new(mesh);

            // Setup parameters
            MR.OffsetParameters op = new();
            op.voxelSize = MR.suggestVoxelSize(mp, 1e6f);

            // Make offset mesh
            var result = MR.offsetMesh(mp, offsetValue, op);

            // Save result
            MR.MeshSave.toAnySupportedFormat(result, "mesh_offset.stl");
        }
        catch (Exception e)
        {
            Console.WriteLine("Error: {0}", e.Message);
        }
    }
}
