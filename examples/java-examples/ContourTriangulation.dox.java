import meshlib.DistanceMap;
import meshlib.Functions;
import meshlib.Image;
import meshlib.Mesh;
import meshlib.Polyline2;

// The class isn't public because of the `.dox.java` file name, the launcher doesn't mind.
class ContourTriangulation
{
    public static void main(String[] args)
    {
        // load image as distance map
        // An error is thrown as `UnexpectedResultException`.
        Image image = meshlib.imageload.Functions.fromAnySupportedFormat("logo.jpg");
        DistanceMap dm = Functions.convertImageToDistanceMap(image, null, null);

        Polyline2 pl2 = Functions.distanceMapTo2DIsoPolyline(dm, 210.0f);
        Mesh triangulationRes = meshlib.planartriangulation.Functions.triangulateContours(pl2.contours(null), null);

        meshlib.meshsave.Functions.toAnySupportedFormat(triangulationRes, "LogoMesh.ctm", null);
    }
}
