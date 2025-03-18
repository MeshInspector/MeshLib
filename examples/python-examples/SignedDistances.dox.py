from meshlib import mrmeshpy

ref_mesh = mrmeshpy.loadMesh("mesh1.ctm")
mesh = mrmeshpy.loadMesh("mesh2.ctm")

# get object of mrmeshpy.VertScalars - set of distances between points of target mesh and reference mesh
a = mrmeshpy.findSignedDistances(ref_mesh, mesh)

print("Id of closest point is " + a.frontId().get())
print("Distance between reference mesh and the closest point of target mesh is " + a.front())
print("Id of farest point is " + a.backId().get())
print("Distance between reference mesh and the farthest point of target mesh is " + a.back())
