from helper import *
from meshlib import mrmeshnumpy
import numpy as np
import unittest as ut
import pytest

# mrmesh uses float32 for vertex coordinates
# however, you could also use float64
def test_numpy_polyline_selfintersections():
    points = np.ndarray(shape=(4,2), dtype=np.float32, buffer=np.array([[0.0,0.0],[1.0,0.0],[1.0,1.0],[0.5,-0.5]], dtype=np.float32))
    polyline = mrmeshnumpy.polyline2FromPoints(points)
    selfInters = mrmesh.findSelfCollidingEdges(polyline)
    assert (selfInters.size() == 1)
    segm1 = mrmesh.LineSegm2f(polyline.orgPnt(mrmesh.EdgeId(selfInters[0].aUndirEdge)),polyline.destPnt(mrmesh.EdgeId(selfInters[0].aUndirEdge)))
    segm2 = mrmesh.LineSegm2f(polyline.orgPnt(mrmesh.EdgeId(selfInters[0].bUndirEdge)),polyline.destPnt(mrmesh.EdgeId(selfInters[0].bUndirEdge)))
    inter = mrmesh.intersection(segm1,segm2)
    assert (abs(inter.x - 2.0/3.0) < 0.001 and inter.y == 0)
    