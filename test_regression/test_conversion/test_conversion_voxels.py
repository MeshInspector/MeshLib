from pathlib import Path
import pytest
import meshlib.mrmeshpy as mlib
from constants import test_files_path
from helpers.meshlib_helpers import compare_voxels


@pytest.mark.parametrize("ext", ["gav"])
def test_conversion_from_raw(ext, tmp_path):
    filename_in = "W48_H58_S78_V1.5e+03_1.5e+03_1.5e+03_G0_F voxels.raw"
    filename_out = "voxels.gav"
    filename_ref = "voxels_ref.gav"

    input_folder = Path(test_files_path) / "conversion" / "voxels_from_raw"

    in_vxl = mlib.loadVoxels(input_folder / filename_in)

    mlib.saveVoxelsGav(in_vxl, str(tmp_path / filename_out))

    compare_voxels(tmp_path / filename_out, input_folder / filename_ref)


@pytest.mark.parametrize("ext", ["gav"])
def test_conversion_to_raw(ext, tmp_path):
    filename_in = "voxels.gav"
    filename_out = "voxels.raw"
    filename_out_full = "W48_H58_S78_V1.5e+03_1.5e+03_1.5e+03_G0_F voxels.raw"


    input_folder = Path(test_files_path) / "conversion" / "voxels_to_raw"

    in_vxl = mlib.loadVoxelsGav(input_folder / filename_in)

    mlib.saveVoxels(in_vxl, str(tmp_path / filename_out))

    saved_vxl = mlib.loadVoxels(tmp_path / filename_out_full)
    compare_voxels(saved_vxl, input_folder / filename_in)
