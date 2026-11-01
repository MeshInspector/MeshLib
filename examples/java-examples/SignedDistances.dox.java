import meshlib.Functions;
import meshlib.Mesh;
import meshlib.VertId;
import meshlib.VertScalars;

// The class isn't public because of the `.dox.java` file name, the launcher doesn't mind.
class SignedDistances
{
    public static void main(String[] args)
    {
        // An error is thrown as `UnexpectedResultException`.
        Mesh refMesh = meshlib.meshload.Functions.fromAnySupportedFormat("mesh1.ctm", null);
        Mesh mesh = meshlib.meshload.Functions.fromAnySupportedFormat("mesh2.ctm", null);

        // get object of VertScalars - set of distances between points of target mesh and reference mesh
        VertScalars vertDistances = Functions.findSignedDistances(refMesh, mesh, null, null);
        float min = Float.MAX_VALUE;
        float max = -Float.MAX_VALUE;
        for (int i = 0; i < vertDistances.size(); i++)
        {
            float d = vertDistances.index(new VertId(i));
            min = Math.min(min, d);
            max = Math.max(max, d);
        }
        System.out.println("Distance between reference mesh and the closest point of target mesh is " + min);
        System.out.println("Distance between reference mesh and the farthest point of target mesh is " + max);
    }
}
