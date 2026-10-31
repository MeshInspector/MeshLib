import meshlib.Functions;
import meshlib.Mesh;
import meshlib.MeshPart;
import meshlib.OffsetParameters;
import meshlib.PointCloud;
import meshlib.UniformSamplingSettings;
import meshlib.Vector3f;
import meshlib.std_optional_MR_Mesh;
import meshlib.std_optional_MR_VertBitSet;

// The class isn't public because of the `.dox.java` file name, the launcher doesn't mind.
class Triangulation
{
    public static void main(String[] args)
    {
        // Generate point cloud
        PointCloud pc = new PointCloud();
        pc.points().reserve(10000);
        for (int i = 0; i < 100; i++)
        {
            double u = 2 * Math.PI * i / (100.0 - 1.0);
            for (int j = 0; j < 100; j++)
            {
                double v = Math.PI * j / (100.0 - 1.0);

                pc.addPoint(new Vector3f(
                    (float)(Math.cos(u) * Math.sin(v)),
                    (float)(Math.sin(u) * Math.sin(v)),
                    (float)Math.cos(v)
                ));
            }
        }
        // Remove duplicate points
        UniformSamplingSettings samplingSettings = new UniformSamplingSettings();
        samplingSettings.setDistance(1e-3f);
        std_optional_MR_VertBitSet vs = Functions.pointUniformSampling(pc, samplingSettings);
        assert vs.has_value();
        pc.setValidPoints(vs.value());
        pc.invalidateCaches();

        // Triangulate it
        std_optional_MR_Mesh triangulated = Functions.triangulatePointCloud(pc, null, null);
        assert triangulated.has_value();

        // Fix possible issues
        MeshPart triangulatedPart = new MeshPart(triangulated.value(), null);
        OffsetParameters offsetParams = new OffsetParameters();
        offsetParams.setVoxelSize(Functions.suggestVoxelSize(triangulatedPart, 5e+6f));
        Mesh mesh = Functions.offsetMesh(triangulatedPart, 0.0f, offsetParams);

        // Save result
        meshlib.meshsave.Functions.toAnySupportedFormat(mesh, "result.stl", null);
    }
}
