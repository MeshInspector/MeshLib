import pytest
from helper import *


def test_fill_hole():
    torus = mrmesh.makeTorus(2, 1, 10, 10, None)

    faceBitSetToDelete = mrmesh.FaceBitSet()
    faceBitSetToDelete.resize(5, False)
    faceBitSetToDelete.set(mrmesh.FaceId(1), True)

    torus.topology.deleteFaces(faceBitSetToDelete)

    holes = torus.topology.findHoleRepresentiveEdges()

    mrmesh.fillHole(torus, holes[0])

    assert torus.topology.findHoleRepresentiveEdges().size() == 0
