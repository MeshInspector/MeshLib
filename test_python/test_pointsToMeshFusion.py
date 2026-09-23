import pytest
from helper import *

try:
    from meshlib import mrcudapy as mrcuda
except ImportError:
    mrcuda = None


def test_point_cloud_triangulation():
    torus_mesh = mrmesh.makeTorus(2, 1, 32, 32, None)
    torus_point_cloud = mrmesh.meshToPointCloud(torus_mesh, True, None)

    params = mrmesh.PointsToMeshParameters()
    params.voxelSize = 0.2
    params.sigma = 0.3
    restored = mrmesh.pointsToMeshFusion(torus_point_cloud, params)

    assert restored.topology.findHoleRepresentiveEdges().size() == 0
    assert abs((42.669 - restored.volume())) < 0.001  # volume assert with some tolerance


def test_point_cloud_triangulation_cuda():
    if mrcuda is None or not hasattr(mrcuda, "ComputePointsToDistanceVolume"):
        pytest.skip("mrcudapy is not built, or is built without voxels support")

    try:  # the bindings unwrap Expected and raise on error
        deviceInfo = mrcuda.getDeviceInfo()
    except Exception as e:
        pytest.skip(f"no CUDA device available: {e}")
    if not deviceInfo.fitForComputations():
        pytest.skip("CUDA device is not fit for computations")

    torus_mesh = mrmesh.makeTorus(2, 1, 32, 32, None)
    torus_point_cloud = mrmesh.meshToPointCloud(torus_mesh, True, None)

    params = mrmesh.PointsToMeshParameters()
    params.voxelSize = 0.2
    params.sigma = 0.3
    params.computeVolume = mrcuda.ComputePointsToDistanceVolume()
    restored = mrmesh.pointsToMeshFusion(torus_point_cloud, params)

    # the same result as the CPU path in test_point_cloud_triangulation
    assert restored.topology.findHoleRepresentiveEdges().size() == 0
    assert abs((42.669 - restored.volume())) < 0.001
