from module_helper import *
from pathlib import Path
from constants import test_files_path
from helpers.meshlib_helpers import compare_voxels

import pytest

@pytest.mark.bindingsV3
def test_points_to_distance_volume(tmp_path):
    input_folder = Path(test_files_path) / "algorithms" / "points_to_dv"

    params = mrmeshpy.PointsToDistanceVolumeParams()
    point_cloud = mrmeshpy.PointsLoad.fromAnySupportedFormat(input_folder / "input.ctm")
    smth = mrmeshpy.pointsToDistanceVolume(point_cloud, params)
    dense_grid = mrmeshpy.simpleVolumeToDenseGrid(smth)
    vdb_volume = mrmeshpy.floatGridToVdbVolume(dense_grid)
    mrmeshpy.VoxelsSave.toAnySupportedFormat(vdb_volume, tmp_path / "output.vdb")

    compare_voxels(input_folder / "output.vdb", tmp_path / "output.vdb")
