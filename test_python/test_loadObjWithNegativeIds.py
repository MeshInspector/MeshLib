import pytest
from helper import *

def test_load_obj_with_negative_ids():
    fs = open("test_obj_neg_ids.obj", "w")
    fs.write("v 0 0 0\n")
    fs.write("v 1 0 0\n")
    fs.write("v 0 1 0\n")
    fs.write("v 1 1 0\n")
    fs.write("f -4 -3 3\n")
    fs.write("f -3 4 -2\n")
    fs.close()
    mesh = mrmesh.loadMesh("test_obj_neg_ids.obj")
    assert mesh.topology.numValidFaces() == 2
    os.remove("test_obj_neg_ids.obj")
