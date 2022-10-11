from helper import *
import pytest


def test_save_mesh():
    torus = mrmesh.makeTorus(2, 1, 10, 10, None)
    smesh = mrmesh.saveMesh(torus, "С:testTorus_dm.stl")
    assert (smesh.has_value())
    os.remove("С:testTorus_dm.stl")