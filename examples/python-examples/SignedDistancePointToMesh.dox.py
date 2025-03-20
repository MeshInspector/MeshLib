from meshlib import mrmeshpy

mesh = mrmeshpy.loadMesh("mesh1.ctm")
point = mrmeshpy.Vector3f(1.5, 1, 0.5)

a = mrmeshpy.findSignedDistance(point, mesh)
print(f"Signed distance between meshes is {a.dist}")
