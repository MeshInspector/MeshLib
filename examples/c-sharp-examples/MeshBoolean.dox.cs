public class MeshBooleanExample
{
    public static void Main(string[] args)
    {
        try
        {
            MR.Mesh mesh_a, mesh_b;
            if (args.Length >= 2)
            {
                // load the two meshes given on the command line
                mesh_a = MR.MeshLoad.fromAnySupportedFormat(args[0]);
                mesh_b = MR.MeshLoad.fromAnySupportedFormat(args[1]);
            }
            else
            {
                // no input given: make two unit spheres shifted along X, so that they overlap
                mesh_a = MR.makeUVSphere(1.0f, 64, 64);
                mesh_b = MR.makeUVSphere(1.0f, 64, 64);
                mesh_b.transform(MR.AffineXf3f.translation(new MR.Vector3f(0.7f, 0.0f, 0.0f)));
            }

            // optional mapper relating the primitives of the input meshes to the primitives of the result
            var mapper = new MR.BooleanResultMapper();
            var parameters = new MR.BooleanParameters();
            parameters.mapper = mapper;

            // perform boolean operation
            MR.BooleanResult res = MR.boolean(mesh_a, mesh_b, MR.BooleanOperation.Intersection, parameters);
            if (!res.valid())
            {
                Console.WriteLine("Error: {0}", res.errorString);
                return;
            }

            // find the faces of the result produced by each input mesh, and the faces the cut created
            var facesOfA = mapper.map(mesh_a.topology.getValidFaces(), MR.BooleanResultMapper.MapObject.A);
            var facesOfB = mapper.map(mesh_b.topology.getValidFaces(), MR.BooleanResultMapper.MapObject.B);
            var newFaces = mapper.newFaces();
            Console.WriteLine("faces from mesh A: {0}", facesOfA.count());
            Console.WriteLine("faces from mesh B: {0}", facesOfB.count());
            Console.WriteLine("faces created by the cut: {0}", newFaces.count());

            // map one particular face of mesh A forward: the cut can split it in several faces of
            // the result, or drop it completely if that part of mesh A is not in the result
            var faceOfA = new MR.FaceId(793);
            var oneFace = new MR.FaceBitSet(794);
            oneFace.set(faceOfA);
            var producedFaces = mapper.map(oneFace, MR.BooleanResultMapper.MapObject.A);
            Console.WriteLine("face {0} of mesh A produced {1} faces of the result", faceOfA.id, producedFaces.count());

            // and backward: the face of mesh A each face of the result came from
            // (invalid id for the faces that came from mesh B)
            if (producedFaces.count() > 0)
            {
                var new2OldFaces = mapper.getNew2OldFaceMap(MR.BooleanResultMapper.MapObject.A);
                var resultFace = producedFaces.find_first();
                Console.WriteLine("face {0} of the result came from face {1} of mesh A",
                    resultFace.id, new2OldFaces[resultFace].id);
            }

            // save result to STL file
            MR.MeshSave.toAnySupportedFormat(res.mesh, "out_boolean.stl");
        }
        catch (Exception e)
        {
            Console.WriteLine("Error: {0}", e.Message);
        }
    }
}
