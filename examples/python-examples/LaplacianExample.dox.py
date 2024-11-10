from meshlib import mrmeshpy as mm
 
# Load mesh
mesh = mm.loadMesh("mesh.stl")
 
# Construct deformer on mesh vertices
lDeformer = mm.Laplacian(mesh)

# Find area for deformation anchor points
ancV0 = mesh.topology.getValidVerts().find_first()
ancV1 = mesh.topology.getValidVerts().find_last()
# Mark anchor points in free area
freeVerts = mm.VertBitSet()
freeVerts.resize(mesh.topology.getValidVerts().size())
freeVerts.set( ancV0, True )
freeVerts.set( ancV1, True )
# Expand free area
mm.expand(mesh.topology,freeVerts,5)

# Initialize laplacian
lDeformer.init(freeVerts,mm.EdgeWeights.CotanWithAreaEqWeight)

shiftAmount = mesh.computeBoundingBox().diagonal()*0.01
# Fix anchor vertices in required position
lDeformer.fixVertex( ancV0, mesh.points.vec[ancV0.get()] + mesh.normal(ancV0) * shiftAmount )
lDeformer.fixVertex( ancV1, mesh.points.vec[ancV1.get()] + mesh.normal(ancV1) * shiftAmount )

# Move free vertices according to anchor ones
lDeformer.apply()

# Invalidate mesh because of external vertices changes
mesh.invalidateCaches()
 
# Save deformed mesh
mm.saveMesh(mesh,"deformed_mesh.stl")
