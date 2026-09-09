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

    # save result to STL file
    mrmeshpy.saveMesh(result.mesh, "out_boolean.stl")
