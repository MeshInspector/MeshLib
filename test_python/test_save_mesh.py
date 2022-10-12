from helper import *
import pytest


def test_save_load_mesh():
    torus = mrmesh.makeTorus(2, 1, 10, 10, None)
    fs = open("test_save_load_mesh.mrmesh", "wb")
    smesh = mrmesh.saveMesh(torus, "*.mrmesh", fs)
    fs.close()
    assert (smesh.has_value())
    fr = open("test_save_load_mesh.mrmesh", "rb")
    torus2 = mrmesh.loadMesh(fr, "*.mrmesh")
    fr.close()
    assert(torus2.has_value())
    assert(torus2.value() == torus)
    os.remove("test_save_load_mesh.mrmesh")