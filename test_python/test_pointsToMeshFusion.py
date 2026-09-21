import pytest
from helper import *

try:
    import meshlib.mrcudapy as mrcuda
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
    if mrcuda is None or not hasattr(mrcuda, "setupPointsToMeshFusion"):
        pytest.skip("mrcudapy is not built, or is built without voxels support")

    torus_mesh = mrmesh.makeTorus(2, 1, 32, 32, None)
    torus_point_cloud = mrmesh.meshToPointCloud(torus_mesh, True, None)

    params = mrmesh.PointsToMeshParameters()
    params.voxelSize = 0.2
    params.sigma = 0.3

    # the callbacks are assignable to the fields, which a plain Python lambda is not
    params.createVolumeCallback = mrcuda.pointsToDistanceVolumeCallback()
    params.createVolumeCallbackByParts = mrcuda.pointsToDistanceVolumeByPartsCallback()

    params.createVolumeCallback = None
    params.createVolumeCallbackByParts = None
    mrcuda.setupPointsToMeshFusion(params)
    assert params.createVolumeCallback is not None
    assert params.createVolumeCallbackByParts is not None

    try:  # the bindings unwrap Expected and raise on error
        deviceInfo = mrcuda.getDeviceInfo()
    except Exception:
        pytest.skip("no CUDA device available")
    if not deviceInfo.fitForComputations():
        pytest.skip("CUDA device is not fit for computations")

    restored = mrmesh.pointsToMeshFusion(torus_point_cloud, params)
    assert restored.topology.findHoleRepresentiveEdges().size() == 0
    assert abs((42.669 - restored.volume())) < 0.001  # same result as on CPU
