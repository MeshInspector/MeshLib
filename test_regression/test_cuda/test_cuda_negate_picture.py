import pytest

from module_helper import *
from pathlib import Path
from pytest_check import check
from constants import test_files_path
from helpers.meshlib_helpers import compare_meshes_similarity, compare_mesh, compare_distance_maps


@pytest.mark.skipif(
    "not config.getoption('--run-cuda')=='positive'",
    reason="Only run when --run-cuda is 'p'",
)
def test_cuda_negate_pic(cuda_module, tmp_path):
    input_folder = Path(test_files_path) / "cuda" / "negate_pic"

    image = mrmeshpy.ImageLoad.fromAnySupportedFormat(input_folder / "input_map.png")
    cuda_module.negatePicture(image)
    mrmeshpy.ImageSave.toAnySupportedFormat(image=image, path=tmp_path / "image_out.png")


    # comparing with Distance Maps is a workaround, but python can't compare images by pixels without heavy
    # external modules, that is too much to include them for this test project for now

    dm_ref = mrmeshpy.loadDistanceMapFromImage(input_folder / "image_out.png")
    dm_out = mrmeshpy.loadDistanceMapFromImage(tmp_path / "image_out.png")
    compare_distance_maps(dm_ref, dm_out)
