import meshlib.FillHoleParams;
import meshlib.Functions;
import meshlib.Mesh;
import meshlib.std_vector_MR_EdgeId;

// The class isn't public because of the `.dox.java` file name, the launcher doesn't mind.
class MeshFillHole
{
    public static void main(String[] args)
    {
        // Load mesh
        Mesh mesh = meshlib.meshload.Functions.fromAnySupportedFormat("mesh.stl", null);

        // Find single edge for each hole in mesh
        std_vector_MR_EdgeId holeEdges = mesh.topology().findHoleRepresentiveEdges(null);

        // Setup filling parameters
        FillHoleParams params = new FillHoleParams();
        params.setMetric(Functions.getUniversalMetric(mesh));

        // Alternatively, Functions.fillHoles(mesh, holeEdges, params) fills all holes at once.
        for (long i = 0; i < holeEdges.size(); i++)
            Functions.fillHole(mesh, holeEdges.at(i), params);

        // Save result
        meshlib.meshsave.Functions.toAnySupportedFormat(mesh, "filledMesh.stl", null);
    }
}
