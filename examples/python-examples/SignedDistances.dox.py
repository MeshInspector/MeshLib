from meshlib import mrmeshpy

ref_mesh = mrmeshpy.loadMesh("mesh1.ctm")
mesh = mrmeshpy.loadMesh("mesh2.ctm")

# get object of mrmeshpy.VertScalars - set of distances between points of target mesh and reference mesh
vert_distances = mrmeshpy.findSignedDistances(ref_mesh, mesh)

# mrmeshpy.VertScalars is iterable so we can use min() or max() functions
print("Distance between reference mesh and the closest point of target mesh is " + str(min(vert_distances)))
print("Distance between reference mesh and the farthest point of target mesh is " + str(max(vert_distances)))
