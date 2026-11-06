public static class MeshBooleanExample
{
    public static void Main(string[] args)
    {
        // create first sphere with radius of 1 unit
        var sphere1 = MR.makeUVSphere(1.0f, 64, 64);

        // create second sphere by cloning the first sphere and moving it in X direction
        var sphere2 = new MR.Mesh(sphere1);
        var xf = MR.AffineXf3f.translation(new MR.Vector3f(0.7f, 0.0f, 0.0f));
        sphere2.transform(xf);

        // optional mapper relating the primitives of the input meshes to the primitives of the result
        MR.BooleanResultMapper mapper = new();
        MR.BooleanParameters parameters = new();
        parameters.mapper = mapper;

        // perform boolean operation
        var result = MR.boolean(sphere1, sphere2, MR.BooleanOperation.Intersection, parameters);
        if (!result.valid())
            Console.Error.WriteLine(result.errorString);

        var resultMesh = result.mesh;

        // find the faces of the result produced by each input sphere, and the faces the cut created
        var facesOfSphere1 = mapper.map(sphere1.topology.getValidFaces(), MR.BooleanResultMapper.MapObject.A);
        var facesOfSphere2 = mapper.map(sphere2.topology.getValidFaces(), MR.BooleanResultMapper.MapObject.B);
        var newFaces = mapper.newFaces();

        Console.WriteLine("faces from sphere1: {0}", facesOfSphere1.count());
        Console.WriteLine("faces from sphere2: {0}", facesOfSphere2.count());
        Console.WriteLine("faces created by the cut: {0}", newFaces.count());

        // map one particular face of sphere1 forward: the cut can split it in several faces of the
        // result, or drop it completely if that part of sphere1 is not in the result
        var faceOfSphere1 = new MR.FaceId(793);
        MR.FaceBitSet oneFace = new();
        oneFace.autoResizeSet(faceOfSphere1);
        var producedFaces = mapper.map(oneFace, MR.BooleanResultMapper.MapObject.A);
        Console.Write("face {0} of sphere1 produced {1} faces of the result:", faceOfSphere1.id, producedFaces.count());
        foreach (var f in producedFaces)
            Console.Write(" {0}", f.id);
        Console.WriteLine();

        // and backward: the face of sphere1 each face of the result came from
        // (invalid id for the faces that came from sphere2)
        var new2OldFaces = mapper.getNew2OldFaceMap(MR.BooleanResultMapper.MapObject.A);
        var resultFace = producedFaces.findFirst();
        Console.WriteLine("face {0} of the result came from face {1} of sphere1", resultFace.id, new2OldFaces[resultFace].id);

        // save result to STL file
        MR.MeshSave.toAnySupportedFormat(resultMesh, "out_boolean.stl");
    }
}
