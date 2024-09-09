from module_helper import *
from pathlib import Path
from pytest_check import check
from constants import test_files_path
from helpers.meshlib_helpers import compare_meshes_similarity, compare_mesh

import pytest


@pytest.mark.smoke
@pytest.mark.parametrize('voxel', [0.2, 0.4])
def test_fusion(tmp_path, voxel):
    """
    Test triangulation algorithm
    """
    #  Load input point
    case_name = f"voxel_{voxel}"
    input_folder = Path(test_files_path) / "algorithms" / "fusion"
    points = mrmeshpy.loadPoints(input_folder / "input.ctm")

    # Make mesh
    params = mrmeshpy.PointsToMeshParameters()
    params.voxelSize = voxel
    params.sigma = 0.8
    params.minWeight = 6
    mesh = mrmeshpy.pointsToMeshFusion(points, params)

    # === Verification
    mrmeshpy.saveMesh(mesh, tmp_path / f"{case_name}.ctm")  # used to store
    ref_mesh_path = input_folder / f"{case_name}.ctm"
    ref_mesh = mrmeshpy.loadMesh(ref_mesh_path)

    with check:
        compare_meshes_similarity(mesh, ref_mesh)
