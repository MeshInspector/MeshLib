from module_helper import *
from pathlib import Path
import pytest
from constants import test_files_path
from helpers.meshlib_helpers import compare_voxels

@pytest.mark.bindingsV3
def test_dcm_folder(i, tmp_path):
    """
    Test the conversion of DICOM files to a voxel representation.
    """
    # Get the path to the DICOM files
    dcm_folder = Path(test_files_path) / "conversion" / "dcm_folder"
    dcm = mrmeshpy.VoxelsLoad.loadDicomsFolderTreeAsVdb(dcm_folder / "input")[0].value()
    mrmeshpy.VoxelsSave.toAnySupportedFormat(dcm.vol, tmp_path / "dcm.vdb")
    compare_voxels(tmp_path / "dcm.vdb", dcm_folder / "dcm.vdb")
