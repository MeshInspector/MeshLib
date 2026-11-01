import meshlib.Functions;
import meshlib.Mesh;
import meshlib.StitchHolesParams;
import meshlib.std_vector_MR_EdgeId;

// The class isn't public because of the `.dox.java` file name, the launcher doesn't mind.
class MeshStitchHole
{
    public static void main(String[] args)
    {
        // Load meshes
        Mesh mesh = meshlib.meshload.Functions.fromAnySupportedFormat("meshAwithHole.stl", null);
        Mesh meshB = meshlib.meshload.Functions.fromAnySupportedFormat("meshBwithHole.stl", null);

        // Unite meshes
        mesh.addMesh(meshB, null, null);

        // Find holes (expect that there are exactly 2 holes)
        std_vector_MR_EdgeId edges = mesh.topology().findHoleRepresentiveEdges(null);
        if (edges.size() != 2)
            System.exit(1);

        // Connect two holes
        StitchHolesParams params = new StitchHolesParams();
        params.setMetric(Functions.getUniversalMetric(mesh));
        Functions.stitchHoles(mesh, edges.front(), edges.back(), params);

        // Save result
        meshlib.meshsave.Functions.toAnySupportedFormat(mesh, "stitchedMesh.stl", null);
    }
}
