from module_helper import *
from pathlib import Path
from pytest_check import check
from constants import test_files_path
from helpers.meshlib_helpers import compare_meshes_similarity, compare_mesh
import meshlib.mrmeshpy as mrmeshpy

import pytest


from module_helper import *
from pathlib import Path
from pytest_check import check
from constants import test_files_path
from helpers.meshlib_helpers import compare_meshes_similarity, compare_mesh
import meshlib.mrmeshpy as mlpy

import pytest


@pytest.mark.parametrize("subdivide_params", [
    {"name": "max_new_verticles",
     "params": {
        "maxEdgeSplits": 2000
     }},
    {"name": "smoothMode",
     "params": {
         "maxEdgeSplits": 2000,
         "smoothMode": True,
     }},
    {"name": "maxAngleChangeAfterFlip",
     "params": {
         "maxEdgeSplits": 2000,
         "maxAngleChangeAfterFlip": 120 * 3.14 / 180,
     }},
    {"name": "maxEdgeLen",
     "params": {
         "maxEdgeLen": 0.1,
     }},
    {"name": "maxTriAspectRatio",
     "params": {
         "maxTriAspectRatio": 3,
         "maxEdgeSplits": 2000,
     }},
    {"name": "maxSplittableTriAspectRatio",
     "params": {
         "maxSplittableTriAspectRatio": 3,
         "maxEdgeSplits": 2000,
     }},
])
def test_subdivide(tmp_path, subdivide_params):
    """
    Test boolean algorithm with all operation types
    """
    #  Load input meshes
    input_folder = Path(test_files_path) / "algorithms" / "subdivide" / "fox"
    case_name = subdivide_params["name"]
    mesh = mlpy.loadMesh(input_folder / "input.mrmesh")

    # Setup decimate parameters
    settings = mlpy.SubdivideSettings()
    for key in subdivide_params["params"].keys():
        settings.__setattr__(key, subdivide_params["params"][key])
    mlpy.subdivideMesh(mesh, settings)

    mlpy.saveMesh(mesh, tmp_path / f"{case_name}.mrmesh")
    # === Verification
    # ref_mesh_path = input_folder / f"{case_name}.mrmesh"
    # ref_mesh = mlpy.loadMesh(ref_mesh_path)
    # #  check meshes similarity (for extra details on fail)
    # with check:
    #     compare_meshes_similarity(mesh, ref_mesh)
    # # check saved file is same as reference
    # with check:
    #     assert compare_mesh(mesh, ref_mesh_path)
