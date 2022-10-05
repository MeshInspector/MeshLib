from helper import *
import pytest

def test_copyMesh():
    torus = mrmesh.makeOuterHalfTorus(2, 1, 10, 10, None)
    
    copyMesh = mrmesh.copyMesh(torus)
    
    assert(copyMesh == torus)