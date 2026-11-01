import meshlib.Functions;
import meshlib.Mesh;
import meshlib.PointCloud;
import meshlib.PointsToMeshParameters;

// The class isn't public because of the `.dox.java` file name, the launcher doesn't mind.
class Fusion
{
    public static void main(String[] args)
    {
        // load points. An error is thrown as `UnexpectedResultException`.
        PointCloud pointCloud = meshlib.pointsload.Functions.fromAnySupportedFormat("Points.ply", null);

        PointsToMeshParameters params = new PointsToMeshParameters();
        params.setVoxelSize(pointCloud.computeBoundingBox(null).diagonal() * 1e-2f);
        params.setSigma(Math.max(params.voxelSize(), Functions.findAvgPointsRadius(pointCloud, 50, null)));
        params.setMinWeight(1.0f);

        Mesh mesh = Functions.pointsToMeshFusion(pointCloud, params);

        meshlib.meshsave.Functions.toAnySupportedFormat(mesh, "Mesh.ctm", null);
    }
}
