public static class MeshOffsetWeightedExample
{
    public static void Main(string[] args)
    {
        // Create some mesh
        var mesh = MR.makeTorus();

        // Create VertScalars obj with weights for every vertex
        var vertSize = mesh.topology.vertSize();
        MR.VertScalars scalars = new(vertSize);
        float maxWeight = 0.0f;
        for (int i = 0; i < (int)vertSize; ++i)
        {
            var v = new MR.VertId(i);
            scalars[v] = MathF.Abs(mesh.points[v].x / 5.0f); // Individual extra offset sizes for points
            maxWeight = MathF.Max(maxWeight, scalars[v]);
        }

        MR.WeightedShell.ParametersMetric parameters = new();
        // Algorithm is voxel based, voxel size affects performance and form of result mesh
        parameters.voxelSize = MR.suggestVoxelSize(mesh, 1000);
        // common basic offset applied for all point
        // Vertex-specific weighted offsets applied after the basic one
        parameters.offset = 0.2f;
        parameters.dist.maxWeight = maxWeight; // should always have maximum between weights provided

        var res = MR.WeightedShell.meshShell(mesh, scalars, parameters);

        MR.MeshSave.toAnySupportedFormat(res, "offset_weighted.ctm");
    }
}
