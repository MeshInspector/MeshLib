import pytest

from constants import test_files_path, DEFAULT_RHAUSDORF_THRESHOLD
from helpers.meshlib_helpers import relative_hausdorff
from module_helper import *
from pathlib import Path


@pytest.mark.smoke
def test_simple_traingulation_sample(tmp_path):
    input_folder = Path(test_files_path) / "doc_samples" / "simple_triangulation"

    # === Sample code
    import numpy as np

    u, v = np.mgrid[0:2 * np.pi:100j, 0:np.pi:100j]
    x = np.cos(u) * np.sin(v)
    y = np.sin(u) * np.sin(v)
    z = np.cos(v)

    # Prepare for MeshLib PointCloud
    verts = np.stack((x.flatten(), y.flatten(), z.flatten()), axis=-1).reshape(-1, 3)
    # Create MeshLib PointCloud from np ndarray
    pc = mrmeshnumpy.pointCloudFromPoints(verts)
    # Remove duplicate points
    if is_new_binding:
        samplingSettings = mrmeshpy.UniformSamplingSettings()
        samplingSettings.distance = 1e-3
        pc.validPoints = mrmeshpy.pointUniformSampling(pc, samplingSettings)
    else:
        pc.validPoints = mrmeshpy.pointUniformSampling(pc, 1e-3)
    pc.invalidateCaches()

    # Triangulate it
    triangulatedPC = mrmeshpy.triangulatePointCloud(pc)

    # Fix possible issues
    offsetSettings = mrmeshpy.OffsetParameters()
    offsetSettings.voxelSize = mrmeshpy.suggestVoxelSize(triangulatedPC, 5e6)
    triangulatedPC = mrmeshpy.offsetMesh(triangulatedPC, 0, offsetSettings)

    #  === Verification
    assert relative_hausdorff(triangulatedPC,
                              input_folder / "trng.stl") > DEFAULT_RHAUSDORF_THRESHOLD
