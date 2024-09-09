from module_helper import *
from pathlib import Path
from pytest_check import check
from constants import test_files_path
from helpers.meshlib_helpers import compare_meshes_similarity, compare_mesh

import pytest


@pytest.mark.smoke
@pytest.mark.parametrize("case", [
    pytest.param({"name": "maxEdgeSplits", "params": {"maxEdgeSplits": 1000}},
                 id="maxEdgeSplits"),
    pytest.param({"name": "finalRelaxIters", "params": {"maxEdgeSplits": 1000, "finalRelaxIters": 1}},
                 id="finalRelaxIters"),
    pytest.param({"name": "maxAngleChangeAfterFlip", "params": {"maxEdgeSplits": 1000, "maxAngleChangeAfterFlip": 50}},
                 id="maxAngleChangeAfterFlip"),
    pytest.param({"name": "maxBdShift", "params": {"maxEdgeSplits": 1000, "maxBdShift": 0.4}},
                 id="maxBdShift"),
])
def test_remesh(tmp_path, case):
    """
    Test remesh algorithm
    """
    #  Load input mesh
    case_name = case["name"]
    input_folder = Path(test_files_path) / "algorithms" / "remesh"
    mesh = mrmeshpy.loadMesh(input_folder / "input.ctm")

    # Remesh
    params = case["params"]
    settings = mrmeshpy.RemeshSettings()
    for key, value in params.items():
        setattr(settings, key, value)

    result = mrmeshpy.remesh(mesh, settings)

    # === Verification
    assert result == True
    assert mesh.topology.findHoleRepresentiveEdges().size() == 0
    mrmeshpy.saveMesh(mesh, tmp_path / f"{case_name}.ctm")  # used to store
    ref_mesh_path = input_folder / f"{case_name}.ctm"
    ref_mesh = mrmeshpy.loadMesh(ref_mesh_path)

    with check:
        compare_meshes_similarity(mesh, ref_mesh)
