import pytest
from helper import *


def test_point_cloud_distance():
    torusMesh = mrmesh.makeTorus(2, 1, 32, 32, None)
    cloud1 = mrmesh.meshToPointCloud(torusMesh, True, None)
    cloud2 = mrmesh.meshToPointCloud(torusMesh, True, None) #same

    dOne = mrmesh.findMaxDistanceSqOneWay(cloud1, cloud2)
    assert dOne == 0

    dBoth = mrmesh.findMaxDistanceSq(cloud1, cloud2)
    assert dBoth == 0
