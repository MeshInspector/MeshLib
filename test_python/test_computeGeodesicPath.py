import pytest
from helper import *


def test_computeGeodesicPath():
    torus = mrmesh.makeTorus(2, 1, 10, 10, None)
    mtp1 = mrmesh.MeshTriPoint(mrmesh.EdgeId(0), mrmesh.TriPointf(0.2, 0.2))
    mtp2 = mrmesh.MeshTriPoint(mrmesh.EdgeId(10), mrmesh.TriPointf(0.2, 0.2))
    try:
        path = mrmesh.computeGeodesicPath(
            torus, mtp1, mtp2, mrmesh.GeodesicPathApprox.DijkstraBiDir
        )
    except ValueError as e:
        print(e)
        assert False
