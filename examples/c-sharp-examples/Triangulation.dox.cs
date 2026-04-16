public class TriangulationExample
{
    public static void Run()
    {
        try
        {
            // Generate point cloud
            MR.PointCloud pc = new();

            const float uConst = MathF.PI * 2 / 100;
            const float vConst = MathF.PI / 101;
            for (int i = 0; i < 100; ++i)
            {
                float u = uConst * i;
                for (int j = 1; j < 100; ++j)
                {
                    float v = vConst * j;

                    pc.addPoint(new(
                        MathF.Cos(u) * MathF.Sin(v),
                        MathF.Sin(u) * MathF.Sin(v),
                        MathF.Cos(v)
                    ));
                }
            }


            // Triangulate it
            MR.TriangulationParameters parameters = new();
            MR.Mesh? triangulated = MR.triangulatePointCloud(pc, parameters).value();
            if (triangulated is null)
            {
                Console.WriteLine("Error during triangulation");
                return;
            }

            // Fix possible issues
            MR.OffsetParameters offsetParameters = new();
            MR.MeshPart mp = new(triangulated);
            offsetParameters.voxelSize = MR.suggestVoxelSize(mp, 5e+6f);
            var offset = MR.offsetMesh(mp, 0f, offsetParameters);

            // Save result
            MR.MeshSave.toAnySupportedFormat(offset, "meshA_icp.stl");
        }
        catch (Exception e)
        {
            Console.WriteLine("Error: {0}", e.Message);
        }
    }
}
