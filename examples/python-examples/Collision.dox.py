from meshlib import mrmeshpy as mm

meshA = mm.makeUVSphere() # make mesh A
meshB = mm.makeUVSphere() # make mesh B
meshB.transform(mm.AffineXf3f.translation(mm.Vector3f(0.1,0.1,0.1))) # shift mesh B for better demonstration

collidingFacePairs = mm.findCollidingTriangles(meshA,meshB) # find each pair of colliding faces
for fp in collidingFacePairs:
    print(fp.aFace,fp.bFace) # print each pair of colliding faces

collidingFaceBitSetA,collidingFaceBitSetB = mm.findCollidingTriangleBitsets(meshA,meshB) # find bitsets of colliding faces
print(collidingFaceBitSetA.count()) # print number of colliding faces from mesh A
print(collidingFaceBitSetB.count()) # print number of colliding faces from mesh B

isColliding = not mm.findCollidingTriangles(meshA,meshB,firstIntersectionOnly=True).empty() # fast check if mesh A and mesh B collide
print(isColliding)
