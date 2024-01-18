from helper import *
import pytest


def test_degenerateBand():
    cube = mrmesh.makeCube()

    faces = mrmesh.FaceBitSet()
    faces.resize( 2, True ) # First two faces

    mrmesh.makeDegenerateBandAroundRegion(cube, faces)

    assert (cube.topology.numValidFaces() == 20)
    assert (cube.area() == mrmesh.makeCube().area())
