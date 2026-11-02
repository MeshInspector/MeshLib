public static class TriangulationExample
{
    public static void Main(string[] args)
    {
        // Generate point cloud
        MR.PointCloud pc = new();
        pc.points.reserve(10000);
        for (int i = 0; i < 100; ++i)
        {
            float u = MathF.PI * 2 * i / (100.0f - 1.0f);
            for (int j = 0; j < 100; ++j)
            {
                float v = MathF.PI * j / (100.0f - 1.0f);

                pc.addPoint(new MR.Vector3f(
                    MathF.Cos(u) * MathF.Sin(v),
                    MathF.Sin(u) * MathF.Sin(v),
                    MathF.Cos(v)
                ));
            }
        }
        // Remove duplicate points
        MR.UniformSamplingSettings samplingSettings = new();
        samplingSettings.distance = 1e-3f;
        var vs = MR.pointUniformSampling(pc, samplingSettings).value()!;
        pc.validPoints.assign(vs);
        pc.invalidateCaches();

        // Triangulate it
        var triangulated = MR.triangulatePointCloud(pc).value()!;

        // Fix possible issues
        MR.OffsetParameters offsetParams = new();
        offsetParams.voxelSize = MR.suggestVoxelSize(triangulated, 5e+6f);
        var mesh = MR.offsetMesh(triangulated, 0.0f, offsetParams);

        // Save result
        MR.MeshSave.toAnySupportedFormat(mesh, "result.stl");
    }
}
