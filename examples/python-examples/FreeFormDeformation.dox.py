from meshlib import mrmeshpy as mm

# Load mesh
mesh = mm.loadMesh("mesh.stl")

# Compute mesh bounding box
box = mesh.computeBoundingBox()

# Construct deformer on mesh vertices
ffDeformer = mm.FreeFormDeformer(mesh.points,mesh.topology.getValidVerts())

# Init deformer with 3x3 grid on mesh box
ffDeformer.init(mm.Vector3i.diagonal(3),box)

# Move some control points of grid to the center
ffDeformer.setRefGridPointPosition(mm.Vector3i(1,1,0),box.center())
ffDeformer.setRefGridPointPosition(mm.Vector3i(1,1,2),box.center())
ffDeformer.setRefGridPointPosition(mm.Vector3i(0,1,1),box.center())
ffDeformer.setRefGridPointPosition(mm.Vector3i(2,1,1),box.center())
ffDeformer.setRefGridPointPosition(mm.Vector3i(1,0,1),box.center())
ffDeformer.setRefGridPointPosition(mm.Vector3i(1,2,1),box.center())

# Apply deformation to mesh vertices
ffDeformer.apply()

# Invalidate mesh because of external vertices changes
mesh.invalidateCaches()

# Save deformed mesh
mm.saveMesh(mesh,"deformed_mesh.stl")
