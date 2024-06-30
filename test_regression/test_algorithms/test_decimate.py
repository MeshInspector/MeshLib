from module_helper import *
from pathlib import Path
from pytest_check import check
from constants import test_files_path
from helpers.meshlib_helpers import compare_meshes_similarity, compare_mesh
import meshlib.mrmeshpy as mlib

import pytest


@pytest.mark.smoke
@pytest.mark.parametrize("dec_params", [
    {"name": "maxError_0.05",
        "params": {
            "maxError": 0.05
        }},
    {"name": "maxError_0.25",
        "params": {
            "maxError": 0.2
        }},
    {"name": "target_triangles_200",
        "params": {
            "maxDeletedFaces": 200,  # tuned to R0003C_V4-16aug19 mesh
            "maxError": 0.05
        }},
    {"name": "maxEdgeLen_245",
        "params": {
            "maxEdgeLen": 245,
            "maxError": 0.03  # tuned to R0003C_V4-16aug19 mesh
        }},
    {"name": "maxEdgeLen_0.4",
        "params": {
            "maxEdgeLen": 0.4,  # tuned to R0003C_V4-16aug19 mesh
            "maxError": 0.05
        }},
    {"name": "maxTriangleAspectRatio_5",
     "params": {
         "maxTriangleAspectRatio": 5,
         "maxError": 0.15
     }},
    {"name": "stabilizer_0.001",
        "params": {
            "stabilizer": 0.001,
            "maxError": 0.03
        }},
    {"name": "stabilizer_0",
        "params": {
            "stabilizer": 0,
            "maxError": 0.03
        }},
    {"name": "strategy_ShortestEdgeFirst",
     "params": {
         "strategy": "ShortestEdgeFirst",
         "maxError": 0.05
     }},
    {"name": "strategy_MinimizeError",
     "params": {
         "strategy": "MinimizeError",
         "maxError": 0.05
     }},
    {"name": "touchBdVertices_true",
     "params": {
         "touchNearBdEdges": True,
         "maxError": 0.05
     }},
    {"name": "touchBdVertices_false",
     "params": {
         "touchNearBdEdges": False,
         "maxError": 0.05
     }},
    {"name": "optimizeVertexPos_false",
     "params": {
         "optimizeVertexPos": False,
         "maxError": 0.05

     }},
    {"name": "optimizeVertexPos_true",
     "params": {
         "optimizeVertexPos": True,
         "maxError": 0.05
     }},
])
def test_decimate(tmp_path, dec_params):
    """
    Test decimate algorithm with all settings, avaliable in UI
    """
    #  Load input meshes
    input_folder = Path(test_files_path) / "algorithms" / "decimate" / "R0003C_V4-16aug19"
    case_name = dec_params["name"]
    mesh = mlib.loadMesh(input_folder / "input.mrmesh")

    # Setup decimate parameters
    settings = mlib.DecimateSettings()
    for key in dec_params["params"].keys():
        if key == "strategy":
            settings.strategy = mlib.DecimateStrategy.__members__[dec_params["params"]["strategy"]]
        else:
            settings.__setattr__(key, dec_params["params"][key])
    settings.packMesh = True
    mlib.decimateMesh(mesh, settings)

    # === Verification
    mlib.saveMesh(mesh, tmp_path / f"{case_name}.ctm")
    ref_mesh_path = input_folder / f"{case_name}.ctm"
    ref_mesh = mlib.loadMesh(ref_mesh_path)
    #  check meshes similarity (for extra details on fail)
    with check:
        compare_meshes_similarity(mesh, ref_mesh,
                                  verts_thresh=0.01)  # diff vs reference usually about 5 verts on 800 overall
    with check:
        self_col_tri = mlib.findSelfCollidingTriangles(mesh).size()
        assert self_col_tri == 0, f"Mesh should have no self-colliding triangles, actual value is {self_col_tri}"
    with check:
        degen_faces = mlib.findDegenerateFaces(mesh).count()
        assert degen_faces == 0, f"Mesh should have no degenerate faces, actual value is {degen_faces}"
