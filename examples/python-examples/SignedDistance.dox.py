from meshlib import mrmeshpy

mesh1 = mrmeshpy.loadMesh("mesh1.ctm")
mesh2 = mrmeshpy.loadMesh("mesh2.ctm")

a = mrmeshpy.findSignedDistance(mesh1, mesh2)
print(f"Signed distance between meshes is {a.signedDist}")
