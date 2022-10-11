from helper import *
import pytest


def test_point_cloud_triangulation():
    torusMesh = mrmesh.makeTorus(2, 1, 32, 32, None)
    torusPointCloud = mrmesh.meshToPointCloud(torusMesh, True, None)

    params = mrmesh.TriangulationParameters()
    restored = mrmesh.triangulatePointCloud(torusPointCloud, params)

    assert (len(restored.points.vec) == 1024)
    assert (restored.topology.getValidVerts().count() == 1024)
    assert (restored.topology.findHoleRepresentiveEdges().size() == 0)
