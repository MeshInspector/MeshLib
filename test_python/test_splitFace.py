import pytest
from helper import *


def test_splitFace():
    R1 = 2
    R2 = 1
    torus = mrmesh.makeTorus(R1, R2, 10, 10, None)

    numFaces = torus.topology.numValidFaces()
    faceMap = mrmesh.FaceHashMap()
    orgFace = mrmesh.FaceId(2)
    torus.splitFace(orgFace, None, faceMap)

    assert torus.topology.numValidFaces() == numFaces + 2
    assert faceMap[mrmesh.FaceId(numFaces)].get() == orgFace.get()
    assert faceMap[mrmesh.FaceId(numFaces + 1)].get() == orgFace.get()
    assert faceMap.size() == 2

    torus.splitEdge(mrmesh.EdgeId(2), None, faceMap)
    assert torus.topology.numValidFaces() == numFaces + 4
    assert faceMap.size() == 4
