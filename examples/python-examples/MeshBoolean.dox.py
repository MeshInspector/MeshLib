import meshlib.mrmeshpy as mrmeshpy

# create first sphere with radius of 1 unit
sphere1 = mrmeshpy.makeUVSphere(1.0, 64, 64)

# create second sphere by cloning the first sphere and moving it in X direction
sphere2 = mrmeshpy.copyMesh(sphere1)
xf = mrmeshpy.AffineXf3f.translation(mrmeshpy.Vector3f(0.7, 0.0, 0.0))
sphere2.transform(xf)

# perform boolean operation
result = mrmeshpy.boolean(sphere1, sphere2, mrmeshpy.BooleanOperation.Intersection)
result_mesh = result.mesh
if not result.valid():
    print(result.errorString)

# save result to STL file
mrmeshpy.saveMesh(result_mesh, "out_boolean.stl")
