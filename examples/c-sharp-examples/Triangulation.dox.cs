using System.Reflection;
using static MR.DotNet;

public class TriangulationExample
{
    public static void Run()
    {
        try
        {
            // Generate point cloud
            Vector3f[] points = new Vector3f[9900];

            const float uConst = MathF.PI * 2 / 100;
            const float vConst = MathF.PI / 101;
            for (int i = 0; i < 100; ++i)
            {
                float u = uConst * i;
                for (int j = 1; j < 100; ++j)
                {
                    float v = vConst * j;

                    points[i * 100 + j] = new Vector3f(
                        MathF.Cos(u) * MathF.Sin(v),
                        MathF.Sin(u) * MathF.Sin(v),
                        MathF.Cos(v)
                    );
                }
            }

            PointCloud pc = PointCloud.FromPoints(points);


            // Triangulate it
            TriangulationParameters parameters = new TriangulationParameters();
            Mesh? triangulated = TriangulatePointCloud(pc, parameters);
            if ( triangulated == null)
            {
                Console.WriteLine("Error during triangulation");
            }

            // Fix possible issues
            OffsetParameters offsetParameters = new OffsetParameters();
            MeshPart mp = new MeshPart(triangulated);
            offsetParameters.voxelSize = Offset.SuggestVoxelSize( mp, 5e+6f);
            Mesh mesh = Offset.OffsetMesh(mp, 0f, offsetParameters);

            // Save result
            MeshSave.ToAnySupportedFormat(mp.mesh, "meshA_icp.stl");
        }
        catch (Exception e)
        {
            Console.WriteLine("Error: {0}", e.Message);
        }
    }
}