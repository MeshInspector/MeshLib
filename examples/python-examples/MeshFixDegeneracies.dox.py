import sys

import meshlib.mrmeshpy as mrmeshpy

# Load mesh
mesh = mrmeshpy.loadMesh("mesh.stl")

# you can set various parameters for the resolving process; see the documentation for more info
params = mrmeshpy.FixMeshDegeneraciesParams()
params.maxDeviation = 1e-5 * mesh.computeBoundingBox().diagonal()
params.tinyEdgeLength = 1e-3

mrmeshpy.fixMeshDegeneracies(mesh, params)

mrmeshpy.saveMesh(mesh, "fixed_mesh.stl")
