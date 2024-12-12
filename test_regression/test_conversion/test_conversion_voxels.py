from module_helper import *
from pathlib import Path
import pytest
from constants import test_files_path
from helpers.meshlib_helpers import compare_voxels


@pytest.mark.parametrize("ext", ["gav", "vdb", "dcm"])
def test_conversion_from_raw(ext, tmp_path):
    filename_in = "W48_H58_S78_V1.5e+03_1.5e+03_1.5e+03_G0_F voxels.raw"
    filename_out = f"voxels.{ext}"
    filename_ref = "voxels_ref.gav"

    input_folder = Path(test_files_path) / "conversion" / "voxels_from_raw"

    vxl_in = mrmeshpy.loadVoxelsRaw(input_folder / filename_in)

    mrmeshpy.saveVoxels(vxl_in, str(tmp_path / filename_out))

    # Verify
    vxl_ref =  mrmeshpy.loadVoxels(input_folder / filename_ref)[0]
    vxl_out =  mrmeshpy.loadVoxels(tmp_path / filename_out)[0]

    compare_voxels(vxl_out, vxl_ref)


@pytest.mark.parametrize("ext", ["gav", "vdb", "dcm"])
def test_conversion_to_raw(ext, tmp_path):
    filename_in = f"voxels.{ext}"
    filename_out = "voxels.raw"
    filename_out_full = "W48_H58_S78_V1.5e+03_1.5e+03_1.5e+03_G0_F voxels.raw"


    input_folder = Path(test_files_path) / "conversion" / "voxels_to_raw"

    vxl_in = mrmeshpy.loadVoxels(input_folder / filename_in)[0]

    mrmeshpy.saveVoxelsRaw(vxl_in, tmp_path / filename_out)

    # Verify
    vxl_out = mrmeshpy.loadVoxels(tmp_path / filename_out_full)[0]
    vxl_ref = mrmeshpy.loadVoxels(input_folder / filename_in)[0]
    compare_voxels(vxl_out, vxl_ref)
