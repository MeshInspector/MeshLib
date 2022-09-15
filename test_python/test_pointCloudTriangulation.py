from helper import *
import pytest

def test_pointCloudTriangulation():
    torusMesh = mrmesh.makeTorus(2,1,32,32,None)
    torusPointCloud = mrmesh.mesh_to_points(torusMesh, True, None)

    params = mrmesh.TriangulationParameters()
    restored = mrmesh.triangulate_point_cloud(torusPointCloud, params)

    assert( len(restored.points.vec) == 1024 )
    assert( restored.topology.getValidVerts().count() == 1024 )
    assert( restored.topology.findHoleRepresentiveEdges().size() == 0)










