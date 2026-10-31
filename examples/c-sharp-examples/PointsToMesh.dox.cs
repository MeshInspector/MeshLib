public static class PointsToMeshExample
{
    public static void Main(string[] args)
    {
        // load points
        var pc = MR.PointsLoad.fromAnySupportedFormat("Points.ply");
        var mesh = MR.triangulatePointCloud(pc).value()!; // can be null only if canceled by progress callback

        MR.MeshSave.toAnySupportedFormat(mesh, "Mesh.ctm");
    }
}
