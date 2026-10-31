public static class PointsToMeshExample
{
    public static void Main(string[] args)
    {
        try
        {
            // load points
            var pc = MR.PointsLoad.fromAnySupportedFormat("Points.ply");
            var mesh = MR.triangulatePointCloud(pc).value()!; // can be null only if canceled by progress callback

            MR.MeshSave.toAnySupportedFormat(mesh, "Mesh.ctm");
        }
        catch (Exception e)
        {
            Console.WriteLine("Error: {0}", e.Message);
        }
    }
}
