using System.Globalization;

public class MeshOffsetExample
{
    public static void Main(string[] args)
    {
        try
        {
            // Make a mesh to offset
            // (to offset your own mesh, load it with MR.MeshLoad.fromAnySupportedFormat("mesh.stl") instead)
            var mesh = MR.makeUVSphere(1.0f, 32, 32);

            // Offset value: the first argument if one is given, otherwise 5% of the bounding box diagonal
            float offsetValue = args.Length > 0
                ? float.Parse(args[0], NumberStyles.AllowDecimalPoint, CultureInfo.InvariantCulture)
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
