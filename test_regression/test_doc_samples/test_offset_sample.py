from constants import test_files_path, DEFAULT_RHAUSDORF_THRESHOLD
from helpers.meshlib_helpers import relative_hausdorff
from module_helper import *
from pathlib import Path


def test_offset_sample(tmp_path):
    input_folder = Path(test_files_path) / "doc_samples" / "offset"

    # === Sample code
    import meshlib.mrmeshpy as mrmeshpy

    # Load mesh
    mesh = mrmeshpy.loadMesh(str(input_folder / "beethoven_in.stl"))

    # Setup parameters
    params = mrmeshpy.OffsetParameters()
    params.voxelSize = mesh.computeBoundingBox().diagonal() * 5e-3  # offset grid precision (algorithm is voxel based)
    if mrmeshpy.findRightBoundary(mesh.topology).empty():
        params.signDetectionMode = mrmeshpy.SignDetectionMode.HoleWindingRule  # use if you have holes in mesh

    # Make offset mesh
    offset = mesh.computeBoundingBox().diagonal() * 0.05
    offsetedMesh = mrmeshpy.offsetMesh(mesh, offset, params)

    # Save result
    mrmeshpy.saveMesh(offsetedMesh, str(tmp_path / "offsetedMesh.stl"))

    #  === Verification
    assert relative_hausdorff(tmp_path / "offsetedMesh.stl",
                              input_folder / "offsetedMesh.stl") > DEFAULT_RHAUSDORF_THRESHOLD
