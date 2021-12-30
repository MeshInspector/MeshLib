from helper import *

def isEqualVector3(a, b):
    diff = a - b
    return diff.length() < 1.e-6


size = mrmesh.Vector3.diagonal( 2 )
pos1 = mrmesh.Vector3.diagonal( 0 )
pos2 = mrmesh.Vector3.diagonal( -1 )
pos3 = mrmesh.Vector3.diagonal( 1 )


# TEST 1

mesh = mrmesh.make_cube(size, pos1)
settings = mrealgorithms.DecimateSettings()

result = mrealgorithms.decimate( mesh, settings )

assert( result.vertsDeleted == 0 )
assert( result.facesDeleted == 0 )
# assert( result.errorIntroduced == 0 )

assert( mesh.topology.getValidVerts().size() == 8 )
assert( mesh.topology.getValidVerts().count() == 8 )
assert( mesh.topology.findHoleRepresentiveEdges().size() == 0 )


# TEST 2

meshA = mrmesh.make_cube(size, pos1)
meshB = mrmesh.make_cube(size, pos2)

bOperation = mrealgorithms.BooleanOperation.Intersection
bResMapper = mrealgorithms.BooleanResultMapper()
bResult = mrealgorithms.boolean( meshA, meshB, bOperation, None, bResMapper )

mesh = bResult.mesh
settings = mrealgorithms.DecimateSettings()

result = mrealgorithms.decimate( mesh, settings )

assert( isEqualVector3( mesh.computeBoundingBox(mesh.topology.getValidFaces(), mrmesh.AffineXf3() ).min , pos1 ) )
assert( isEqualVector3( mesh.computeBoundingBox(mesh.topology.getValidFaces(), mrmesh.AffineXf3() ).max , pos3 ) )

assert( result.vertsDeleted == 6 )
assert( result.facesDeleted == 12 )
# assert( result.errorIntroduced == 0 )

assert( mesh.topology.getValidVerts().size() == 14 )
assert( mesh.topology.getValidVerts().count() == 8 )
assert( mesh.topology.findHoleRepresentiveEdges().size() == 0 )
