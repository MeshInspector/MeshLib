from helper import *
import mrmeshnumpy
import numpy as np
import unittest as ut
import pytest

# mrmesh uses float32 for vertex coordinates
# however, you could also use float64
def test_numpy_pointsbuild1():
    verts = np.ndarray(shape=(4,3), dtype=np.float32, buffer=np.array([[0.0,0.0,0.0],[1.0,0.0,0.0],[1.0,1.0,0.0],[0.0,1.0,0.0]], dtype=np.float32))

    pc = mrmeshnumpy.pointCloudFromPoints(verts)
    assert (pc.validPoints.count() == 4)
    assert (pc.normals.vec.size() == 0)

# mrmesh uses float32 for vertex coordinates
# however, you could also use float64

    norms = np.ndarray(shape=(4,3), dtype=np.float32, buffer=np.array([[0.0,0.0,1.0],[0.0,0.0,1.0],[0.0,0.0,1.0],[0.0,0.0,1.0]], dtype=np.float32))

    pc = mrmeshnumpy.pointCloudFromPoints(verts, norms)
    assert (pc.validPoints.count() == 4)
    assert (pc.normals.vec.size() == 4)