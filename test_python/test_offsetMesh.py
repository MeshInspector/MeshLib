from helper import *
import pytest


def test_offset_mesh():
    cube = mrmesh.makeCube()
    expCubeOffset = mrmesh.offsetMesh(cube,0.1)
    assert (expCubeOffset.has_value())
    assert (expCubeOffset.value().volume()/cube.volume() > 1.0)
