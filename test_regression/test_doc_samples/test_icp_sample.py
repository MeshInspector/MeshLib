from constants import test_files_path, DEFAULT_RHAUSDORF_THRESHOLD
from helpers.meshlib_helpers import relative_hausdorff
from module_helper import *
from pathlib import Path


def test_icp_sample(tmp_path):
    input_folder = Path(test_files_path) / "doc_samples" / "icp"

    # === Sample code

    import meshlib.mrmeshpy as mrmeshpy

    # Load meshes
    meshFloating = mrmeshpy.loadMesh(str(input_folder / "beethoven_moved.stl"))
    meshFixed = mrmeshpy.loadMesh(str(input_folder / "beethoven_in.stl"))

    # Prepare ICP parameters
    diagonal = meshFixed.getBoundingBox().diagonal()
    icpSamplingVoxelSize = diagonal * 0.01  # To sample points from object
    icpParams = mrmeshpy.ICPProperties()
    icpParams.distThresholdSq = (diagonal * 0.1) ** 2  # Select points pairs that's not too far
    icpParams.exitVal = diagonal * 0.003  # Stop when this distance reached

    # Calculate transformation
    icp = mrmeshpy.ICP(meshFloating, meshFixed,
                       mrmeshpy.AffineXf3f(), mrmeshpy.AffineXf3f(),
                       icpSamplingVoxelSize)
    icp.setParams(icpParams)
    icp.updatePointPairs()
    xf = icp.calculateTransformation()

    # Transform floating mesh
    meshFloating.transform(xf)

    # Output information string
    print(icp.getLastICPInfo())

    # Save result
    mrmeshpy.saveMesh(meshFloating, str(tmp_path / "meshA_icp.stl"))

    #  === Verification
    assert relative_hausdorff(tmp_path / "meshA_icp.stl",
                              input_folder / "meshA_icp.stl") > DEFAULT_RHAUSDORF_THRESHOLD
