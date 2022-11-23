from helper import *
from meshlib import mrmeshnumpy
import numpy as np
import unittest as ut
import pytest

def test_numpy_meshbuild():
    faces = np.ndarray(shape=(2,3), dtype=np.int32, buffer=np.array([[0,1,2],[2,3,0]], dtype=np.int32))

    # mrmesh uses float32 for vertex coordinates
    # however, you could also use float64
    verts = np.ndarray(shape=(4,3), dtype=np.float32, buffer=np.array([[0.0,0.0,0.0],[1.0,0.0,0.0],[1.0,1.0,0.0],[0.0,1.0,0.0]], dtype=np.float32))

    mesh = mrmeshnumpy.meshFromFacesVerts(faces, verts)

    assert (mesh.topology.getValidFaces().count() == 2)
    a = mrmesh.VertId(0)
    b = mrmesh.VertId(0)
    c = mrmesh.VertId(0)
    mesh.topology.getTriVerts(mrmesh.FaceId(0),a,b,c)
    assert (a.get() == 0)
    assert (b.get() == 1)
    assert (c.get() == 2)

    assert (mesh.points.vec.size() == 4)
    np.testing.assert_almost_equal (mesh.points.vec[0].z, 0.0)
    np.testing.assert_almost_equal (mesh.points.vec[2].x, 1.0)
