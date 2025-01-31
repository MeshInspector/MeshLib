from module_helper import *
from pathlib import Path
from pytest_check import check
from constants import test_files_path
from helpers.meshlib_helpers import compare_meshes_similarity

import pytest

@pytest.mark.smoke
def test_points_to_distance_volume(tmp_path):
    input_folder = Path(test_files_path) / "algorithms" / "distance_map"

    params = mrmeshpy.PointsToDistanceVolumeParams()
    point_cloud = mrmeshpy.PointsLoad.fromAnySupportedFormat("c:\Work\_meshes\Torus Points.ctm")
    smth = mrmeshpy.pointsToDistanceVolume(point_cloud, params)
    dense_grid = mrmeshpy.simpleVolumeToDenseGrid(smth)
    vdb_volume = mrmeshpy.floatGridToVdbVolume(dense_grid)
    mrmeshpy.VoxelsSave.toAnySupportedFormat(vdb_volume, "a.vdb")
