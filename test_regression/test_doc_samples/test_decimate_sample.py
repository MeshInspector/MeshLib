from module_helper import *
from pathlib import Path
from constants import test_files_path, DEFAULT_RHAUSDORF_THRESHOLD
from helpers.meshlib_helpers import relative_hausdorff


def test_decimate_sample(tmp_path):
    input_folder = Path(test_files_path) / "doc_samples" / "decimate"

    # === Sample code
    import meshlib.mrmeshpy as mrmeshpy

    # Load mesh
    mesh = mrmeshpy.loadMesh(str(Path(input_folder) / "beethoven_in.stl"))

    # Setup decimate parameters
    settings = mrmeshpy.DecimateSettings()
    settings.maxError = 0.05

    # Decimate mesh
    mrmeshpy.decimateMesh(mesh, settings)

    # Save result
    mrmeshpy.saveMesh(mesh, str(tmp_path / "decimatedMesh.stl"))

    #  === Verification
    assert relative_hausdorff(tmp_path / "decimatedMesh.stl",
                              input_folder / "decimatedMesh.stl") > DEFAULT_RHAUSDORF_THRESHOLD
