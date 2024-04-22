from module_helper import *
from pathlib import Path
from pytest_check import check
from constants import test_files_path
from helpers.meshlib_helpers import compare_meshes_similarity, compare_mesh
import meshlib.mrmeshpy as mlib

import pytest


@pytest.mark.smoke
@pytest.mark.parametrize("neighbors", [16, 8])
def test_triangulation(tmp_path, neighbors):
    """
    Test triangulation algorithm
    """
    #  Load input point
    case_name = f"neighbors_{neighbors}"
    input_folder = Path(test_files_path) / "algorithms" / "triangulation"
    points = mlib.loadPoints(input_folder / "input.ctm")

    # Traingulate points
    params = mlib.TriangulationParameters()
    params.numNeighbours = 16
    params.critAngle = 90
    mesh = mlib.triangulatePointCloud(pointCloud=points)

    # === Verification
    mlib.saveMesh(mesh, tmp_path / f"{case_name}.ctm")  # used to store
    ref_mesh_path = input_folder / f"{case_name}.ctm"
    ref_mesh = mlib.loadMesh(ref_mesh_path)

    with check:
        compare_meshes_similarity(mesh, ref_mesh)
