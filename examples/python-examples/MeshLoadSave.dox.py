import meshlib.mrmeshpy as mrmeshpy

try:
    mesh = mrmeshpy.loadMesh("mesh.stl")
except ValueError as e:
    print(e)

mrmeshpy.saveMesh(mesh, "mesh.ply")
