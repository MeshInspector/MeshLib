import meshlib.AffineXf3f;
import meshlib.ICP;
import meshlib.ICPProperties;
import meshlib.Mesh;
import meshlib.MeshOrPoints;

// The class isn't public because of the `.dox.java` file name, the launcher doesn't mind.
class MeshICP
{
    public static void main(String[] args)
    {
        // Load meshes. An error is thrown as `UnexpectedResultException`.
        Mesh meshFloating = meshlib.meshload.Functions.fromAnySupportedFormat("meshA.stl", null);
        Mesh meshFixed = meshlib.meshload.Functions.fromAnySupportedFormat("meshB.stl", null);

        // Prepare ICP parameters
        float diagonal = meshFixed.getBoundingBox().diagonal();
        float icpSamplingVoxelSize = diagonal * 0.01f; // To sample points from object
        ICPProperties icpParams = new ICPProperties();
        icpParams.setDistThresholdSq((diagonal * 0.1f) * (diagonal * 0.1f)); // Use points pairs with maximum distance specified
        icpParams.setExitVal(diagonal * 0.003f); // Stop when distance reached

        // Calculate transformation
        ICP icp = new ICP(
            new MeshOrPoints(meshFloating),
            new MeshOrPoints(meshFixed),
            new AffineXf3f(), new AffineXf3f(),
            icpSamplingVoxelSize);
        icp.setParams(icpParams);
        AffineXf3f xf = icp.calculateTransformation();

        // Transform floating mesh
        meshFloating.transform(xf, null);

        // Output information string
        String info = icp.getStatusInfo().toString();
        System.err.println(info);

        // Save result
        meshlib.meshsave.Functions.toAnySupportedFormat(meshFloating, "meshA_icp.stl", null);
    }
}
