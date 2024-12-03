using System.Globalization;
using System.Reflection;
using static MR.DotNet;

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
            MeshPart mp = new MeshPart(MeshLoad.FromAnySupportedFormat("mesh.stl"));

            // Setup parameters
            OffsetParameters op = new OffsetParameters();
            op.voxelSize = Offset.SuggestVoxelSize(mp, 1e6f);

            // Make offset mesh
            var result = Offset.OffsetMesh(mp, offsetValue, op);

            // Save result
            MeshSave.ToAnySupportedFormat(result, "mesh_offset.stl");
        }
        catch (Exception e)
        {
            Console.WriteLine("Error: {0}", e.Message);
        }
    }
}
