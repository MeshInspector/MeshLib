from module_helper import *
from pathlib import Path
from pytest_check import check
from constants import test_files_path
from helpers.meshlib_helpers import compare_meshes_similarity, compare_mesh

import pytest


@pytest.mark.smoke
@pytest.mark.parametrize("subdivide_params", [
    pytest.param({"name": "max_new_verticles",
     "params": {
        "maxEdgeSplits": 2000
     }},id="max_new_verticles"),
    pytest.param({"name": "smoothMode",
     "params": {
         "maxEdgeSplits": 2000,
         "smoothMode": True,
     }},id="smoothMode"),
    pytest.param({"name": "maxAngleChangeAfterFlip",
     "params": {
         "maxEdgeSplits": 2000,
         "maxAngleChangeAfterFlip": 120 * 3.14 / 180,
     }},id="maxAngleChangeAfterFlip"),
    pytest.param({"name": "maxEdgeLen",
     "params": {
         "maxEdgeLen": 0.1,
     }},id="maxEdgeLen"),
    pytest.param({"name": "maxTriAspectRatio",
     "params": {
         "maxTriAspectRatio": 3,
         "maxEdgeSplits": 2000,
     }},id="maxTriAspectRatio"),
    pytest.param({"name": "maxSplittableTriAspectRatio",
     "params": {
         "maxSplittableTriAspectRatio": 3,
         "maxEdgeSplits": 2000,
     }},id="maxSplittableTriAspectRatio"),
])
def test_subdivide(tmp_path, subdivide_params):
    """
    Test subdivide algorithm with all settings, available in UI
    """
    #  Load input meshes
    input_folder = Path(test_files_path) / "algorithms" / "subdivide" / "fox"
    case_name = subdivide_params["name"]
    mesh = mrmeshpy.loadMesh(input_folder / "input.mrmesh")

    # Setup decimate parameters
    settings = mrmeshpy.SubdivideSettings()
    for key in subdivide_params["params"].keys():
        settings.__setattr__(key, subdivide_params["params"][key])
    mrmeshpy.subdivideMesh(mesh, settings)

    # === Verification
    mrmeshpy.saveMesh(mesh, tmp_path / f"{case_name}.ctm")  # used to store
    ref_mesh_path = input_folder / f"{case_name}.ctm"
    ref_mesh = mrmeshpy.loadMesh(ref_mesh_path)
    #  check meshes similarity (for extra details on fail)
    with check:
        compare_meshes_similarity(mesh, ref_mesh)
    with check:
        self_col_tri = mrmeshpy.findSelfCollidingTriangles(mesh).size()
        assert self_col_tri == 0, f"Mesh should have no self-colliding triangles, actual value is {self_col_tri}"
    with check:
        degen_faces = mrmeshpy.findDegenerateFaces(mesh).count()
        assert degen_faces == 0, f"Mesh should have no degenerate faces, actual value is {degen_faces}"
