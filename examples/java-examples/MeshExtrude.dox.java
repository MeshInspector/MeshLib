import meshlib.FaceBitSet;
import meshlib.FaceId;
import meshlib.Functions;
import meshlib.Mesh;
import meshlib.Vector3f;
import meshlib.VertBitSet;
import meshlib.VertId;

// The class isn't public because of the `.dox.java` file name, the launcher doesn't mind.
class MeshExtrude
{
    public static void main(String[] args)
    {
        // Load mesh. An error is thrown as `UnexpectedResultException`.
        Mesh mesh = meshlib.meshload.Functions.fromAnySupportedFormat("mesh.stl", null);

        // Select faces to extrude
        FaceBitSet facesToExtrude = new FaceBitSet();
        facesToExtrude.autoResizeSet(new FaceId(1), true);
        facesToExtrude.autoResizeSet(new FaceId(2), true);

        // Create duplicated verts on region boundary
        Functions.makeDegenerateBandAroundRegion(mesh, facesToExtrude, null);

        // Find vertices that will be moved
        VertBitSet vertsForMove = Functions.getIncidentVerts(mesh.topology(), facesToExtrude);

        for (VertId v = vertsForMove.find_first(); v.valid(); v = vertsForMove.find_next(v))
        {
            // Move each vertex
            Functions.opAddAssign(mesh.points().index_mut(v), Vector3f.plusZ());
        }

        // Invalidate internal caches after manual changing
        mesh.invalidateCaches(null);

        // Save mesh
        meshlib.meshsave.Functions.toAnySupportedFormat(mesh, "extrudedMesh.stl", null);
    }
}
