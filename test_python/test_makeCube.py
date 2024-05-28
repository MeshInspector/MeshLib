import pytest
from helper import *


def test_make_cube():
    size = mrmesh.Vector3f.diagonal(2)
    pos1 = mrmesh.Vector3f.diagonal(1)
    pos2 = mrmesh.Vector3f.diagonal(3)

    cube = mrmesh.makeCube(size, pos1)
    assert cube.topology.numValidVerts() == 8
    assert cube.topology.numValidFaces() == 12

    cube2 = mrmesh.makeCube(size, pos2)

    transVector = mrmesh.Vector3f()
    transVector.x = 0.5
    transVector.y = 1
    transVector.z = 1
    diffXf = mrmesh.AffineXf3f.translation(transVector)
    cube2.transform(diffXf)

    xf = mrmesh.AffineXf3f()
    cube1 = cube
    pairs = mrmesh.findCollidingTriangles(cube1, cube2, None, False)
    assert len(pairs) == 0

    cube.topology.flipOrientation()
