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

    mrmesh.makeBridgeEdge(t, mrmesh.EdgeId(10), mrmesh.EdgeId(60))

    assert t.edgeSize() - edges_num_before == 2, "Function should add some edges"
