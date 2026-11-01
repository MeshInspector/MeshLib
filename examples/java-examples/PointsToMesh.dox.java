import meshlib.Functions;
import meshlib.PointCloud;
import meshlib.std_optional_MR_Mesh;

// The class isn't public because of the `.dox.java` file name, the launcher doesn't mind.
class PointsToMesh
{
    public static void main(String[] args)
    {
        // load points. An error is thrown as `UnexpectedResultException`.
        PointCloud pointCloud = meshlib.pointsload.Functions.fromAnySupportedFormat("Points.ply", null);

        std_optional_MR_Mesh triangulationRes = Functions.triangulatePointCloud(pointCloud, null, null);
        assert triangulationRes.has_value(); // can be empty only if canceled by progress callback

        meshlib.meshsave.Functions.toAnySupportedFormat(triangulationRes.value(), "Mesh.ctm", null);
    }
}
