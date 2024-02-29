from module_helper import *
from pathlib import Path
from pytest_check import check
from constants import test_files_path
from helpers.meshlib_helpers import compare_meshes_similarity, compare_mesh
import meshlib.mrmeshpy as mlib

import pytest


@pytest.mark.parametrize("metric", ["getMinAreaMetric", "getUniversalMetric",
                                    "getEdgeLengthStitchMetric", "getCircumscribedMetric"])
@pytest.mark.parametrize("input", ["torus", "crocodile"])
def test_fill_holes(tmp_path, input, metric):
    """
    Test fill holes algorithm
    """
    #  Load input meshes
    input_folder = Path(test_files_path) / "algorithms" / "fill" / input
    case_name = f"{input}_{metric}"
    mesh = mlib.loadMesh(input_folder / "input.mrmesh")

    # Find holes
    edges = mesh.topology.findHoleRepresentiveEdges()

    # Connect two holes
    params = mlib.FillHoleParams()
    params.metric = getattr(mlib, metric)(mesh)
    params.maxPolygonSubdivisions = 100
    mlib.fillHole(mesh, edges[0], params)

    # === Verification
    mlib.saveMesh(mesh, tmp_path / f"{case_name}.mrmesh")  # used to store
    ref_mesh_path = input_folder / f"{case_name}.mrmesh"
    ref_mesh = mlib.loadMesh(ref_mesh_path)

    with check:
        compare_meshes_similarity(mesh, ref_mesh)
    with check:
        degen_faces = mlib.findDegenerateFaces(mesh).count()
        assert degen_faces == 0, f"Mesh should have no degenerate faces, actual value is {degen_faces}"

