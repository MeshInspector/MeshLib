from helper import *


def test_makeBridgeEdge():
    torus = mrmesh.makeTorus(2, 1, 10, 10, None)

    faceBitSetToDelete = mrmesh.FaceBitSet()
    faceBitSetToDelete.resize(5, False)
    faceBitSetToDelete.set(mrmesh.FaceId(1), True)
    faceBitSetToDelete.set(mrmesh.FaceId(11), True)

    torus.topology.deleteFaces(faceBitSetToDelete)

    t = torus.topology
    edges_num_before = t.edgeSize()
    edges = torus.topology.findHoleRepresentiveEdges()

    mrmesh.makeBridgeEdge(t, edges[0], edges[1])

    assert t.edgeSize() - edges_num_before == 2, "Function should add some edges"
