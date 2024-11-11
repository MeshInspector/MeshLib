import meshlib.mrmeshpy as mrmeshpy

# Load meshes
mesh_a = mrmeshpy.loadMesh("meshAwithHole.stl")
mesh_b = mrmeshpy.loadMesh("meshBwithHole.stl")

# Unite meshes
mesh = mrmeshpy.mergeMeshes([mesh_a, mesh_b])

# Find holes
edges = mesh.topology.findHoleRepresentiveEdges()

# Connect two holes
params = mrmeshpy.StitchHolesParams()
params.metric = mrmeshpy.getUniversalMetric(mesh)
mrmeshpy.buildCylinderBetweenTwoHoles(mesh, edges[0], edges[1], params)

# Save result
mrmeshpy.saveMesh(mesh, "stitchedMesh.stl")
