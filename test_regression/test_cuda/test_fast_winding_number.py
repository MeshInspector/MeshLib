import pytest

from module_helper import *
from pathlib import Path
from pytest_check import check
from constants import test_files_path
from helpers.meshlib_helpers import compare_meshes_similarity, compare_mesh

@pytest.mark.skipif(
    "not config.getoption('--run-cuda')=='positive'",
    reason="Only run when --run-cuda is 'p'",
)
def test_cuda_mesh_to_dm(cuda_module, tmp_path):
    input_folder = Path(test_files_path) / "cuda" / "offset"
    mesh = mrmeshpy.loadMesh(input_folder / "input.ctm")
    offset_params = mrmeshpy.GeneralOffsetParameters()
    offset_params.signDetectionMode = mrmeshpy.SignDetectionMode.HoleWindingRule
    offset_params.fwn = cuda_module.FastWindingNumber(mesh)  # Enables usage of CUDA
    offset_params.voxelSize = mrmeshpy.suggestVoxelSize(mesh, 5e6)

    offset_mesh = mrmeshpy.generalOffsetMesh(mp=mesh, offset=0, params=offset_params)
    mrmeshpy.saveMesh(offset_mesh, tmp_path / "cuda_offset.ctm")

    with check:
        compare_meshes_similarity(tmp_path / "cuda_offset.ctm", input_folder / "cuda_offset.ctm")
