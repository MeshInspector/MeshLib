import meshlib.AffineXf3f;
import meshlib.BooleanOperation;
import meshlib.BooleanResult;
import meshlib.BooleanResultMapper;
import meshlib.BooleanResultMapper_MapObject;
import meshlib.FaceBitSet;
import meshlib.FaceId;
import meshlib.FaceMap;
import meshlib.Functions;
import meshlib.Mesh;
import meshlib.Vector3f;

// The class isn't public because of the `.dox.java` file name, the launcher doesn't mind.
class MeshBoolean
{
    public static void main(String[] args)
    {
        // First, create a unit sphere.
        Mesh sphere1 = Functions.makeUVSphere(1.0f, 64, 64);

        // Create a copy of this sphere and offset it.
        Mesh sphere2 = new Mesh(sphere1);
        AffineXf3f xf = AffineXf3f.translation(new Vector3f(0.7f, 0.0f, 0.0f));
        sphere2.transform(xf, null);

        // Ask for an optional mapper relating the primitives of the input meshes to the primitives of the result.
        BooleanResultMapper mapper = new BooleanResultMapper();

        // Perform the boolean operation.
        BooleanResult result = Functions.boolean_(sphere1, sphere2, BooleanOperation.Intersection, null, mapper, null);
        if (!result.valid())
        {
            System.err.println("Failed to perform boolean: " + result.errorString());
            System.exit(1);
        }

        // Find the faces of the result produced by each input sphere, and the faces the cut created.
        FaceBitSet facesOfSphere1 = mapper.map(sphere1.topology().getValidFaces(), BooleanResultMapper_MapObject.A);
        FaceBitSet facesOfSphere2 = mapper.map(sphere2.topology().getValidFaces(), BooleanResultMapper_MapObject.B);
        FaceBitSet newFaces = mapper.newFaces();

        System.out.println("faces from sphere1: " + facesOfSphere1.count());
        System.out.println("faces from sphere2: " + facesOfSphere2.count());
        System.out.println("faces created by the cut: " + newFaces.count());

        // Map one particular face of sphere1 forward: the cut can split it in several faces of the
        // result, or drop it completely if that part of sphere1 is not in the result.
        FaceId faceOfSphere1 = new FaceId(793);
        FaceBitSet oneFace = new FaceBitSet();
        oneFace.autoResizeSet(faceOfSphere1, true);
        FaceBitSet producedFaces = mapper.map(oneFace, BooleanResultMapper_MapObject.A);
        System.out.println("face " + faceOfSphere1.id_() + " of sphere1 produced " + producedFaces.count() + " faces of the result");

        // And backward: the face of sphere1 each face of the result came from
        // (invalid id for the faces that came from sphere2).
        FaceMap new2oldFaces = mapper.getNew2OldFaceMap(BooleanResultMapper_MapObject.A);
        FaceId resultFace = producedFaces.find_first();
        System.out.println("face " + resultFace.id_() + " of the result came from face " + new2oldFaces.index(resultFace).id_() + " of sphere1");

        // Save result to an STL file. An error is thrown as `UnexpectedResultException`.
        meshlib.meshsave.Functions.toAnySupportedFormat(result.mesh(), "out_boolean.stl", null);
    }
}
