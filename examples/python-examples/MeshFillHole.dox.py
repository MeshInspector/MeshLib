import meshlib.mrmeshpy as mrmeshpy

# Load mesh
mesh = mrmeshpy.loadMesh("mesh.stl")

# Find single edge for each hole in mesh
hole_edges = mesh.topology.findHoleRepresentiveEdges()

for e in hole_edges:
    #  Setup filling parameters
    params = mrmeshpy.FillHoleParams()
    params.metric = mrmeshpy.getUniversalMetric(mesh)
    #  Fill hole represented by `e`
    mrmeshpy.fillHole(mesh, e, params)

# Save result
mrmeshpy.saveMesh(mesh, "filledMesh.stl")
