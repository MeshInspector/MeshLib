from helper import *
import pytest

def test_triangulateContours():

    c2f = mrmesh.Contour2f()
    c2f.resize(5)
    c2f[0].x = 0
    c2f[0].y = 0
    c2f[1].x = 1
    c2f[1].y = 0
    c2f[2].x = 1
    c2f[2].y = 1
    c2f[3].x = 0
    c2f[3].y = 1
    c2f[4] = c2f[0]
    cs2f = mrmesh.Contours2f()
    cs2f.append(c2f)
    mesh = mrmesh.triangulateContours(cs2f)
    
    assert(mesh.topology.numValidFaces() == 2)
    
 
    