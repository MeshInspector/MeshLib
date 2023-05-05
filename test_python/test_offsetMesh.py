from helper import *
import pytest


def test_offset_mesh():
    cube = mrmesh.makeCube()
    try:
        cube_offset = mrmesh.offsetMesh(cube, 0.1)
    except ValueError as e:
        print(e)
        assert False

    assert cube_offset.volume() / cube.volume() > 1.0
