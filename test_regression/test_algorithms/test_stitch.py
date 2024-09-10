from module_helper import *
from pathlib import Path
from pytest_check import check
from constants import test_files_path
from helpers.meshlib_helpers import compare_meshes_similarity, compare_mesh

import pytest


@pytest.mark.smoke
@pytest.mark.parametrize("metric", ["getMinAreaMetric", "getUniversalMetric",
                                    "getEdgeLengthStitchMetric", "getCircumscribedMetric"])
@pytest.mark.parametrize("input", ["torus", "crocodile"])
def test_stitch(tmp_path, input, metric):
    """
    Test stitch holes algorithm
    """
    #  Load input meshes
    input_folder = Path(test_files_path) / "algorithms" / "stitch" / input
    case_name = f"{input}_{metric}"
    mesh = mrmeshpy.loadMesh(input_folder / "input.ctm")

    # Find holes
    edges = mesh.topology.findHoleRepresentiveEdges()

    # Connect two holes
    params = mrmeshpy.StitchHolesParams()
    params.metric = getattr(mrmeshpy, metric)(mesh)
    mrmeshpy.buildCylinderBetweenTwoHoles(mesh, edges[0], edges[1], params)

    # === Verification
    mrmeshpy.saveMesh(mesh, tmp_path / f"{case_name}.ctm")  # used to store
    ref_mesh_path = input_folder / f"{case_name}.ctm"
    ref_mesh = mrmeshpy.loadMesh(ref_mesh_path)

    with check:
        compare_meshes_similarity(mesh, ref_mesh)
