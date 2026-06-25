from meshlib import mrmeshpy as mm

meshA = mm.makeUVSphere() # make mesh A
meshB = mm.makeUVSphere() # make mesh B
meshB.transform(mm.AffineXf3f.translation(mm.Vector3f(0.1,0.1,0.1))) # shift mesh B for better demonstration

converters = mm.getVectorConverters(meshA,meshB) # create converters to integer field (needed for absolute precision predicates)
collidingFaceEdges = mm.findCollidingEdgeTrisPrecise(meshA,meshB,converters.toInt) # find each intersecting edge/triangle pair 
# print pairs of edges tirangles
for vet in collidingFaceEdges:
    if vet.isEdgeATriB():
        print("edgeA:",vet.edge.get(),"triB:",vet.tri().get())
    else:
        print("triA:",vet.tri().get(),"edgeB:",vet.edge.get())
