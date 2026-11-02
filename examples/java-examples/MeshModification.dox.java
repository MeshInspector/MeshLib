import meshlib.AffineXf3f;
import meshlib.Functions;
import meshlib.Matrix3f;
import meshlib.Mesh;
import meshlib.MeshRelaxParams;
import meshlib.SubdivideSettings;
import meshlib.Vector3f;

// The class isn't public because of the `.dox.java` file name, the launcher doesn't mind.
class MeshModification
{
    public static void main(String[] args)
    {
        Mesh mesh = Functions.makeTorus(null, null, null, null, null);

        // Relax mesh (5 iterations)
        MeshRelaxParams relaxParams = new MeshRelaxParams();
        relaxParams.setIterations(5);
        Functions.relax(mesh, relaxParams, null);

        // Subdivide mesh
        SubdivideSettings props = new SubdivideSettings();
        props.setMaxDeviationAfterFlip(0.5f);
        Functions.subdivideMesh(mesh, props);

        // Rotate mesh
        AffineXf3f rotationXf = AffineXf3f.linear(Matrix3f.rotation(Vector3f.plusZ(), (float)Math.PI * 0.5f));
        mesh.transform(rotationXf, null);
    }
}
