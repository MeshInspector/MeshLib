import meshlib.EdgeWeights;
import meshlib.Functions;
import meshlib.Laplacian;
import meshlib.Mesh;
import meshlib.VertBitSet;
import meshlib.VertId;
import meshlib.VertexMass;

// The class isn't public because of the `.dox.java` file name, the launcher doesn't mind.
class LaplacianDeformation
{
    public static void main(String[] args)
    {
        // Load mesh. An error is thrown as `UnexpectedResultException`.
        Mesh mesh = meshlib.meshload.Functions.fromAnySupportedFormat("mesh.stl", null);

        // Construct deformer on the mesh vertices
        Laplacian lDeformer = new Laplacian(mesh);

        // Find an area for the deformation anchor points
        VertId ancV0 = mesh.topology().getValidVerts().find_first();
        VertId ancV1 = mesh.topology().getValidVerts().find_last();
        // Mark the anchor points in the free area
        VertBitSet freeVerts = new VertBitSet();
        freeVerts.resize(mesh.topology().getValidVerts().size(), false);
        freeVerts.set(ancV0, true);
        freeVerts.set(ancV1, true);
        // Expand the free area
        Functions.expand(mesh.topology(), freeVerts, 5);

        // Initialize laplacian
        lDeformer.init(freeVerts, EdgeWeights.Cotan, VertexMass.NeiArea, null);

        float shiftAmount = mesh.computeBoundingBox(null).diagonal() * 0.01f;
        // Fix the anchor vertices in the required position
        lDeformer.fixVertex(ancV0, Functions.opAdd(mesh.points().index(ancV0), Functions.opMul(mesh.normal(ancV0), shiftAmount)), null);
        lDeformer.fixVertex(ancV1, Functions.opAdd(mesh.points().index(ancV1), Functions.opMul(mesh.normal(ancV1), shiftAmount)), null);

        // Move the free vertices according to the anchor ones
        lDeformer.apply();

        // Invalidate the mesh because of the external vertex changes
        mesh.invalidateCaches(null);

        // Save the deformed mesh
        meshlib.meshsave.Functions.toAnySupportedFormat(mesh, "deformed_mesh.stl", null);
    }
}
