using System.Globalization;

public class MeshOffsetExample
{
    public static void Main(string[] args)
    {
        try
        {
            // Load the mesh given as the first argument, or make a sphere to offset if no path is given
            var mesh = args.Length > 0
                ? MR.MeshLoad.fromAnySupportedFormat(args[0])
                : MR.makeUVSphere(1.0f, 32, 32);

            // Offset value: the second argument if given, otherwise 5% of the bounding box diagonal
            float offsetValue = args.Length > 1
                ? float.Parse(args[1], NumberStyles.AllowDecimalPoint, CultureInfo.InvariantCulture)
                : mesh.computeBoundingBox().diagonal() * 0.05f;

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
