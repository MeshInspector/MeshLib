from module_helper import *
from pathlib import Path
from pytest_check import check
from constants import test_files_path
from helpers.meshlib_helpers import compare_meshes_similarity, compare_mesh

import pytest


@pytest.mark.smoke
@pytest.mark.parametrize("input_case", ["sphere", "fox"])
@pytest.mark.parametrize("operation_type", (x for x in mrmeshpy.BooleanOperation.__members__.keys() if x != "Count"))
def test_boolean(tmp_path, operation_type, input_case):
    """
    Test boolean algorithm with all operation types
    """
    #  Load input meshes
    input_folder = Path(test_files_path) / "algorithms" / "boolean" / input_case
    mesh1 = mrmeshpy.loadMesh(input_folder / "meshA.ctm")
    mesh2 = mrmeshpy.loadMesh(input_folder / "meshB.ctm")

    # perform boolean operation
    result = mrmeshpy.boolean(mesh1, mesh2, mrmeshpy.BooleanOperation.__members__[operation_type])
    assert result.valid(), result.errorString
    result_mesh = result.mesh

    # === Verification
    mrmeshpy.saveMesh(result_mesh, tmp_path / f"{operation_type}.ctm")
    ref_mesh_path = input_folder / f"bool_{operation_type}.ctm"
    ref_mesh = mrmeshpy.loadMesh(ref_mesh_path)
    # no self colliding triangles
    with check:
        assert mrmeshpy.findSelfCollidingTriangles(result_mesh).size() == 0
    #  check meshes similarity (for extra details on fail)
    with check:
        compare_meshes_similarity(ref_mesh, result_mesh)
