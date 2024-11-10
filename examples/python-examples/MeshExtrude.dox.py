import meshlib.mrmeshpy as mrmeshpy

# Load mesh
mesh = mrmeshpy.loadMesh("mesh.stl")

# Prepare region to extrude
faces_to_extrude = mrmeshpy.FaceBitSet()
faces_to_extrude.resize(3, False)
faces_to_extrude.set(mrmeshpy.FaceId(1), True)
faces_to_extrude.set(mrmeshpy.FaceId(2), True)

# Create duplicated verts on region boundary
mrmeshpy.makeDegenerateBandAroundRegion(mesh, faces_to_extrude)

# Find vertices that will be moved
verts_for_move = mrmeshpy.getIncidentVerts(mesh.topology, faces_to_extrude)

# Move each vertex
for v in range(verts_for_move.size()):
    if verts_for_move.test(mrmeshpy.VertId(v)):
        mesh.points.vec[v] += mrmeshpy.Vector3f(0.0, 0.0, 1.0)

# Invalidate internal caches after manual changing
mesh.invalidateCaches()

# Save mesh
mrmeshpy.saveMesh(mesh, "extrudedMesh.stl")
