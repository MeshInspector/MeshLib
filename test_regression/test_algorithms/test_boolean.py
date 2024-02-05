from helpers.file_helpers import is_file_same_as_reference
from module_helper import *
from pathlib import Path
from pytest_check import check
from constants import test_files_path
from helpers.meshlib_helpers import compare_meshes_similarity
import meshlib.mrmeshpy as mrmeshpy

import pytest


@pytest.mark.parametrize("input_case", ["sphere", "fox"])
@pytest.mark.parametrize("operation_type", mrmeshpy.BooleanOperation.__members__.keys())
def test_boolean(tmp_path, operation_type, input_case):
    """
    Test boolean algorithm with all operation types
    """
    #  Load input meshes
    input_folder = Path(test_files_path) / "algorithms" / "boolean" / input_case
    mesh1 = mrmeshpy.loadMesh(input_folder / "meshA.mrmesh")
    mesh2 = mrmeshpy.loadMesh(input_folder / "meshB.mrmesh")

    # perform boolean operation
    result = mrmeshpy.boolean(mesh1, mesh2, mrmeshpy.BooleanOperation.__members__[operation_type])
    assert result.valid(), result.errorString
    result_mesh = result.mesh

    # === Verification
    mrmesh.saveMesh(result_mesh, tmp_path / f"bool_{operation_type}.mrmesh")
    ref_mesh = mrmeshpy.loadMesh(input_folder / f"bool_{operation_type}.mrmesh")
    # no self colliding triangles
    with check:
        assert mrmeshpy.findSelfCollidingTriangles(result_mesh).size() == 0
    #  check meshes similarity (for extra details on fail)
    with check:
        compare_meshes_similarity(ref_mesh, result_mesh)
    # check saved file is same as reference
    with check:
        mrmesh.saveMesh(result_mesh, tmp_path / f"bool_{operation_type}.mrmesh")
        is_file_same_as_reference(tmp_path / f"bool_{operation_type}.mrmesh",
                                  input_folder / f"bool_{operation_type}.mrmesh")
