import meshlib.mrmeshpy as mrmeshpy

# create first sphere with radius of 1 unit
sphere1 = mrmeshpy.makeUVSphere(1.0, 64, 64)

# create second sphere by cloning the first sphere and moving it in X direction
sphere2 = mrmeshpy.copyMesh(sphere1)
xf = mrmeshpy.AffineXf3f.translation(mrmeshpy.Vector3f(0.7, 0.0, 0.0))
sphere2.transform(xf)

# optional mapper relating the primitives of the input meshes to the primitives of the result
mapper = mrmeshpy.BooleanResultMapper()

# perform boolean operation
result = mrmeshpy.boolean(sphere1, sphere2, mrmeshpy.BooleanOperation.Intersection, None, mapper)
if not result.valid():
    print(result.errorString)
else:
    # find the faces of the result produced by each input sphere, and the faces the cut created
    map_object = mrmeshpy.BooleanResultMapper.MapObject
    faces_of_sphere1 = mapper.map(sphere1.topology.getValidFaces(), map_object.A)
    faces_of_sphere2 = mapper.map(sphere2.topology.getValidFaces(), map_object.B)
    new_faces = mapper.newFaces()
    print(f"faces from sphere1: {faces_of_sphere1.count()}")
    print(f"faces from sphere2: {faces_of_sphere2.count()}")
    print(f"faces created by the cut: {new_faces.count()}")

    # map one particular face of sphere1 forward: the cut can split it in several faces of the
    # result, or drop it completely if that part of sphere1 is not in the result
    face_of_sphere1 = mrmeshpy.FaceId(793)
    one_face = mrmeshpy.FaceBitSet()
    one_face.autoResizeSet(face_of_sphere1)
    produced_faces = mapper.map(one_face, map_object.A)
    print(f"face {face_of_sphere1.get()} of sphere1 produced {produced_faces.count()} faces of the result")

    # and backward: the face of sphere1 each face of the result came from
    # (invalid id for the faces that came from sphere2)
    new2old_faces = mapper.getNew2OldFaceMap(map_object.A)
    result_face = produced_faces.find_first()
    print(f"face {result_face.get()} of the result came from face {new2old_faces[result_face].get()} of sphere1")

    # save result to STL file
    mrmeshpy.saveMesh(result.mesh, "out_boolean.stl")
