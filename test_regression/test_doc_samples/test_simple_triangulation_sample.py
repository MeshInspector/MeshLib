from constants import test_files_path, DEFAULT_RHAUSDORF_THRESHOLD
from helpers.meshlib_helpers import relative_hausdorff
from module_helper import *
from pathlib import Path


def test_simple_traingulation_sample(tmp_path):
    input_folder = Path(test_files_path) / "doc_samples" / "simple_triangulation"

    # === Sample code
    from meshlib import mrmeshpy as mm
    from meshlib import mrmeshnumpy as mn
    import numpy as np

    u, v = np.mgrid[0:2 * np.pi:100j, 0:np.pi:100j]
    x = np.cos(u) * np.sin(v)
    y = np.sin(u) * np.sin(v)
    z = np.cos(v)

    # Prepare for MeshLib PointCloud
    verts = np.stack((x.flatten(), y.flatten(), z.flatten()), axis=-1).reshape(-1, 3)
    # Create MeshLib PointCloud from np ndarray
    pc = mn.pointCloudFromPoints(verts)
    # Remove duplicate points
    pc.validPoints = mm.pointUniformSampling(pc, 1e-3)
    pc.invalidateCaches()

    # Triangulate it
    triangulatedPC = mm.triangulatePointCloud(pc)

    # Fix possible issues
    triangulatedPC = mm.offsetMesh(triangulatedPC, 0.0)

    #  === Verification
    assert relative_hausdorff(triangulatedPC,
                              input_folder / "trng.stl") > DEFAULT_RHAUSDORF_THRESHOLD
