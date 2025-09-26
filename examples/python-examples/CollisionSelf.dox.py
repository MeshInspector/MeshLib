from meshlib import mrmeshpy as mm

mesh = mm.makeTorusWithSelfIntersections() # make torus with self-intersections

selfCollidingParis = mm.findSelfCollidingTriangles(mesh) # find self-intersecting faces pairs
for fp in selfCollidingParis:
    print(fp.aFace,fp.bFace) # print each pair

selfCollidingBitSet = mm.findSelfCollidingTrianglesBS(mesh) # find bitset of self-intersecting faces
print(selfCollidingBitSet.count()) # print number of self-intersecting faces

isSelfColliding = mm.findSelfCollidingTriangles(mesh,None) # fast check if mesh has self-intersections
print(isSelfColliding)
