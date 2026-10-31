import meshlib.Functions;
import meshlib.GeneralOffsetParameters;
import meshlib.Mesh;
import meshlib.MeshPart;
import meshlib.SignDetectionMode;

// The class isn't public because of the `.dox.java` file name, the launcher doesn't mind.
class MeshOffset
{
    public static void main(String[] args)
    {
        // Create mesh
        Mesh mesh = Functions.makeCube(null, null);
        MeshPart meshPart = new MeshPart(mesh, null);

        // Setup parameters
        GeneralOffsetParameters params = new GeneralOffsetParameters();
        // calculate voxel size depending on desired accuracy and/or memory consumption
        params.setVoxelSize(Functions.suggestVoxelSize(meshPart, 10000000.0f));
        if (!Functions.findRightBoundary(mesh.topology(), null, null).empty())
            params.setSignDetectionMode(SignDetectionMode.HoleWindingRule); // use if you have holes in mesh

        // Make offset mesh
        float offset = mesh.computeBoundingBox(null).diagonal() * 0.1f;
        // An error is thrown as `UnexpectedResultException`.
        Mesh result = Functions.generalOffsetMesh(meshPart, offset, params);

        // Save result
        meshlib.meshsave.Functions.toAnySupportedFormat(result, "mesh_offset.stl", null);
    }
}
