import pytest
from pytest_check import check

from module_helper import *
from pathlib import Path
from constants import test_files_path
from helpers.meshlib_helpers import compare_distance_maps


@pytest.mark.skipif(
    "not config.getoption('--run-cuda')=='positive'",
    reason="Only run when --run-cuda is 'positive'",
)
def test_dm_from_contour(cuda_module, tmp_path):
    #  Load input point
    input_folder = Path(test_files_path) / "cuda" / "lines_to_dm"
    pl3 = mrmeshpy.loadLines(input_folder / "input.mrlines")
    pl2 = mrmeshpy.Polyline2(pl3.contours2())

    # process
    params = mrmeshpy.ContourToDistanceMapParams()
    params.pixelSize = mrmeshpy.Vector2f(0.1, 0.1)
    params.resolution = mrmeshpy.Vector2i(100, 100)
    params.orgPoint = mrmeshpy.Vector2f(0, -5)
    dm = cuda_module.distanceMapFromContours(pl2, params)
    mrmeshpy.saveDistanceMapToImage(dm, tmp_path / "dm_out.png")

    # verification

    dm_ref = mrmeshpy.loadDistanceMapFromImage(input_folder / "dm_out.png")
    dm_out = mrmeshpy.loadDistanceMapFromImage(tmp_path / "dm_out.png")
    with check:
        assert 40000 < cuda_module.distanceMapFromContoursHeapBytes(pl2, params) < 80000
    with check:
        compare_distance_maps(dm_ref, dm_out)
