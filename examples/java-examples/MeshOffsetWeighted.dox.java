import meshlib.Functions;
import meshlib.Mesh;
import meshlib.MeshPart;
import meshlib.VertId;
import meshlib.VertScalars;
import meshlib.weightedshell.ParametersMetric;

// The class isn't public because of the `.dox.java` file name, the launcher doesn't mind.
class MeshOffsetWeighted
{
    public static void main(String[] args)
    {
        // Create some mesh
        Mesh mesh = Functions.makeTorus(null, null, null, null, null);

        // Create VertScalars obj with weights for every vertex
        long vertSize = mesh.topology().vertSize();
        VertScalars scalars = new VertScalars(vertSize);
        float maxWeight = 0.0f;
        for (int i = 0; i < vertSize; i++)
        {
            VertId v = new VertId(i);
            float weight = Math.abs(mesh.points().index(v).x() / 5.0f); // Individual extra offset sizes for points
            scalars.autoResizeSet(v, weight);
            maxWeight = Math.max(maxWeight, weight);
        }

        ParametersMetric params = new ParametersMetric();
        // Algorithm is voxel based, voxel size affects performance and form of result mesh
        params.setVoxelSize(Functions.suggestVoxelSize(new MeshPart(mesh, null), 1000.0f));
        // common basic offset applied for all point
        // Vertex-specific weighted offsets applied after the basic one
        params.setOffset(0.2f);
        params.dist().setMaxWeight(maxWeight); // should always have maximum between weights provided

        // An error is thrown as `UnexpectedResultException`.
        Mesh result = meshlib.weightedshell.Functions.meshShell(mesh, scalars, params);

        meshlib.meshsave.Functions.toAnySupportedFormat(result, "offset_weighted.ctm", null);
    }
}
