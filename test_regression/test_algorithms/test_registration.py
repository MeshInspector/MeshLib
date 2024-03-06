from pathlib import Path

from pytest_check import check

from helpers.meshlib_helpers import compare_meshes_similarity
from module_helper import *
import meshlib.mrmeshpy as mlib
from constants import test_files_path
import pytest


@pytest.mark.parametrize("input_case", [
    {"case_name": "RigidScale", "input_mesh": "fox_geometrik_moved_rotated_scaled.mrmesh", "method": "RigidScale"},
    {"case_name": "AnyRigidXf", "input_mesh": "fox_geometrik_moved_rotated.mrmesh", "method": "AnyRigidXf"},
    {"case_name": "AnyRigidXf_rot-only", "input_mesh": "fox_geometrik_rotated.mrmesh", "method": "AnyRigidXf"},
    {"case_name": "TranslationOnly ", "input_mesh": "fox_geometrik_moved.mrmesh", "method": "TranslationOnly"},
    {"case_name": "OrthogonalAxis", "input_mesh": "fox_geometrik_rotated.mrmesh", "method": "OrthogonalAxis"},
    {"case_name": "FixedAxis", "input_mesh": "fox_geometrik_rotated.mrmesh", "method": "FixedAxis"},
    ]
)
def test_ICP(tmp_path, input_case):
    input_folder = Path(test_files_path) / "algorithms" / "icp"
    case_name = input_case["case_name"]
    mesh_floating = mlib.loadMesh(input_folder / input_case["input_mesh"])
    mesh_fixed = mlib.loadMesh(input_folder / "fox_geometrik.mrmesh")

    # Prepare ICP parameters
    diagonal = mesh_fixed.getBoundingBox().diagonal()
    icp_sampling_voxel_size = diagonal * 0.01  # To sample points from object
    icp_params = mlib.ICPProperties()
    icp_params.distTresholdSq = (diagonal * 0.1) ** 2  # Select points pairs that's not too far
    icp_params.exitVal = diagonal * 0.003  # Stop when this distance reached
    icp_params.icpMode = mlib.ICPMode.__members__[input_case["method"]]  # Select ICP method

    # Calculate transformation
    icp = mlib.ICP(mesh_floating, mesh_fixed,
                       mlib.AffineXf3f(), mlib.AffineXf3f(),
                       icp_sampling_voxel_size)
    icp.setParams(icp_params)
    icp.updateVertPairs()
    xf = icp.calculateTransformation()

    # Transform floating mesh
    mesh_floating.transform(xf)

    # Output information string
    print(icp.getLastICPInfo())

    # Verification
    mlib.saveMesh(mesh_floating, tmp_path / f"{case_name}.mrmesh")
    with check:
        compare_meshes_similarity(mesh_fixed, mesh_floating, rhsdr_thresh=0.1, vol_thresh=0.1, area_thresh=0.1,
                                  verts_thresh=0)
    with check:
        degen_faces = mlib.findDegenerateFaces(mesh_floating).count()
        assert degen_faces == 0, f"Mesh should have no degenerate faces, actual value is {degen_faces}"
    with check:
        self_col_tri = mlib.findSelfCollidingTriangles(mesh_floating).size()
        assert self_col_tri == 0, f"Mesh should have no self-colliding triangles, actual value is {self_col_tri}"


