import meshlib.Functions;
import meshlib.Mesh;
import meshlib.PointCloud;
import meshlib.PointsLoadSettings;
import meshlib.SaveSettings;
import meshlib.VertColors;

// The class isn't public because of the `.dox.java` file name, the launcher doesn't mind.
class TerrainTriangulation
{
    public static void main(String[] args)
    {
        // load points
        VertColors colors = new VertColors();
        PointsLoadSettings pls = new PointsLoadSettings();
        pls.setColors(colors);
        // An error is thrown as `UnexpectedResultException`.
        PointCloud pointCloud = meshlib.pointsload.Functions.fromAnySupportedFormat("TerrainPoints.ply", pls);
        Mesh triangulationRes = Functions.delaunayTriangulationXY_const_MR_PointCloud_ref(pointCloud, null);

        SaveSettings ss = new SaveSettings();
        if (pointCloud.points().size() == colors.size())
            ss.setColors(colors);
        meshlib.meshsave.Functions.toAnySupportedFormat(triangulationRes, "TerrainMesh.ctm", ss);
    }
}
