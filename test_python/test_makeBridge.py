from helper import *


def test_makeBridgeEdge():
    torus = mrmesh.makeTorus(2, 1, 10, 10, None)

    face_bit_set_to_delete = mrmesh.FaceBitSet()
    face_bit_set_to_delete.resize(12, False)
    face_bit_set_to_delete.set(mrmesh.FaceId(1), True)
    face_bit_set_to_delete.set(mrmesh.FaceId(11), True)

    torus.topology.deleteFaces(face_bit_set_to_delete)

    t = torus.topology
    faces_num_before = t.numValidFaces()

    edges = torus.topology.findHoleRepresentiveEdges()

    mrmesh.makeBridge(t, edges[0], edges[1])

    assert t.numValidFaces() - faces_num_before == 2, "Function should add some faces"

def test_makeBridgeEdge():
    torus = mrmesh.makeTorus(2, 1, 10, 10, None)

    face_bit_set_to_delete = mrmesh.FaceBitSet()
    face_bit_set_to_delete.resize(12, False)
    face_bit_set_to_delete.set(mrmesh.FaceId(1), True)
    face_bit_set_to_delete.set(mrmesh.FaceId(11), True)

    torus.topology.deleteFaces(face_bit_set_to_delete)

    t = torus.topology
    edges_num_before = t.edgeSize()
    edges = torus.topology.findHoleRepresentiveEdges()

    mrmesh.makeBridgeEdge(t, edges[0], edges[1])

    assert t.edgeSize() - edges_num_before == 2, "Function should add some edges"
