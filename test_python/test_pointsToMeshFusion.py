import pytest
from helper import *


def test_point_cloud_triangulation():
    torus_mesh = mrmesh.makeTorus(2, 1, 32, 32, None)
    torus_point_cloud = mrmesh.meshToPointCloud(torus_mesh, True, None)

    params = mrmesh.PointsToMeshParameters()
    params.voxelSize = 0.2
    params.sigma = 0.3
    restored = mrmesh.pointsToMeshFusion(torus_point_cloud, params)

    assert restored.topology.findHoleRepresentiveEdges().size() == 0
    assert abs((42.669 - restored.volume())) < 0.001  # volume assert with some tolerance
