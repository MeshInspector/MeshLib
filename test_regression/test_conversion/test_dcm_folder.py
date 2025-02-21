from module_helper import *
from pathlib import Path
import pytest
from constants import test_files_path
from helpers.meshlib_helpers import compare_voxels

@pytest.mark.bindingsV3
def test_dcm_folder(tmp_path):
    """
    Test the conversion of DICOM files to a voxel representation.
    """
    # Get the path to the DICOM files
    dcm_folder = Path(test_files_path) / "conversion" / "dcm_folder"
    dcm = mrmeshpy.VoxelsLoad.loadDicomsFolderTreeAsVdb(dcm_folder / "input")[0].value()
    assert dcm, "bool(dcm) should return True on success loading"
    mrmeshpy.VoxelsSave.toAnySupportedFormat(dcm.vol, tmp_path / "dcm.vdb")
    compare_voxels(tmp_path / "dcm.vdb", dcm_folder / "dcm.vdb")

@pytest.mark.bindingsV3
def test_dcm_folder_negative():
    """
    Test the conversion of DICOM files to a voxel representation.
    """
    # Get the path to the DICOM files
    dcm_folder = Path(test_files_path) / "conversion" / "dcm_folder"
    dcm = mrmeshpy.VoxelsLoad.loadDicomsFolderTreeAsVdb(dcm_folder / "input2")[0]
    assert not dcm, "bool(dcm) should return False on failed loading"
    assert dcm.error() # non empty error string
