public static class FusionExample
{
    public static void Main(string[] args)
    {
        try
        {
            // load points
            var pc = MR.PointsLoad.fromAnySupportedFormat("Points.ply");

            MR.PointsToMeshParameters parameters = new();
            parameters.voxelSize = pc.computeBoundingBox().diagonal() * 1e-2f;
            parameters.sigma = MathF.Max(parameters.voxelSize, MR.findAvgPointsRadius(pc, 50));
            parameters.minWeight = 1.0f;

            var mesh = MR.pointsToMeshFusion(pc, parameters);

            MR.MeshSave.toAnySupportedFormat(mesh, "Mesh.ctm");
        }
        catch (Exception e)
        {
            Console.WriteLine("Error: {0}", e.Message);
        }
    }
}
