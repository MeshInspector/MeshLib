from helper import *
import pytest

def test_computePerFaceNormals():
    torus = mrmesh.makeTorus(2, 1, 10, 10, None)
    
    normals = mrmesh.computePerFaceNormals(torus)