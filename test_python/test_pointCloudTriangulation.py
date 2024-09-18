import pytest
from helper import *


def test_point_cloud_triangulation():
    torusMesh = mrmesh.makeTorus(2, 1, 32, 32, None)
    torusPointCloud = mrmesh.meshToPointCloud(torusMesh, True, None)

    params = mrmesh.TriangulationParameters()
    restored = mrmesh.triangulatePointCloud(torusPointCloud, params)

    assert len(restored.points.vec) == 1024
    assert restored.topology.getValidVerts().count() == 1024
    assert restored.topology.findHoleRepresentiveEdges().size() == 0

def test_create_normals():
    torusMesh = mrmesh.makeTorus(2, 1, 32, 32, None)
    torusPointCloud = mrmesh.meshToPointCloud(torusMesh, False, None)
    settings = mrmesh.TriangulationHelpersSettings()
    settings.numNeis = 16

    allLocal = mrmesh.buildUnitedLocalTriangulations(torusPointCloud, settings)
    unoriented = mrmesh.makeUnorientedNormals(torusPointCloud,allLocal,None,mrmesh.OrientNormals.TowardOrigin)
    oriented = mrmesh.makeOrientedNormals(torusPointCloud,allLocal)

    assert unoriented.vec.size() == torusPointCloud.points.vec.size()
    assert oriented.vec.size() == torusPointCloud.points.vec.size()