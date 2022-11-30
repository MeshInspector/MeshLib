from helper import *
import pytest


def test_distance_map():
    R1 = 2
    R2 = 1
    torus = mrmesh.makeTorus(R1, R2, 10, 10, None)

    params = mrmesh.MeshToDistanceMapParams()
    params.resolution.x = 20
    params.resolution.y = 20
    params.useDistanceLimits = False
    params.allowNegativeValues = False

    params.xRange.x = 2 * (R1 + R2)
    params.yRange.y = 2 * (R1 + R2)
    params.direction.z = 1

    params.orgPoint.x = -(R1 + R2)
    params.orgPoint.y = -(R1 + R2)
    params.orgPoint.z = -R2

    map = mrmesh.computeDistanceMapD(mrmesh.MeshPart(torus), params)
    
    tomesh = mrmesh.distanceMapToMesh(map, mrmesh.DistanceMapToWorld(params))

    assert (map.isValid(0,0) == False)
    assert (map.isValid(7,7) == True)
    assert (map.isValid(9,9) == False)
    assert (map.isValid(10,10) == False)
    assert (map.isValid(13,13) == True)
    assert (map.isValid(19,19) == False)
    assert (tomesh.topology.numValidFaces()>0)