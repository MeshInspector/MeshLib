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

                    pc.AddPoint(new(
                        MathF.Cos(u) * MathF.Sin(v),
                        MathF.Sin(u) * MathF.Sin(v),
                        MathF.Cos(v)
                    ));
                }
            }


            // Triangulate it
            MR.TriangulationParameters parameters = new();
            MR.Mesh? triangulated = MR.TriangulatePointCloud(pc, parameters).Value.Value();
            if (triangulated is null)
            {
                Console.WriteLine("Error during triangulation");
                return;
            }

            // Fix possible issues
            MR.OffsetParameters offsetParameters = new();
            MR.MeshPart mp = new(triangulated);
            offsetParameters.VoxelSize = MR.SuggestVoxelSize(mp, 5e+6f);
            MR.Expected_MRMesh_StdString offset_ex = MR.OffsetMesh(mp, 0f, offsetParameters);
            if (offset_ex.GetError() is var offset_error and not null)
                throw new Exception(offset_error);

            // Save result
            MR.Expected_Void_StdString save_ex = MR.MeshSave.ToAnySupportedFormat(offset_ex.GetValue()!, "meshA_icp.stl");
            if (save_ex.GetError() is var save_error and not null)
                throw new Exception(save_error);
        }
        catch (Exception e)
        {
            Console.WriteLine("Error: {0}", e.Message);
        }
    }
}
