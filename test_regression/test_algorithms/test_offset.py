import pytest

from helpers.meshlib_helpers import compare_meshes_similarity
from module_helper import *
import meshlib.mrmeshpy as mlib
from pytest_check import check
from pathlib import Path
from constants import test_files_path


@pytest.mark.parametrize("test_params", [
    {"name": "thicken_open_vdb",
     "mesh": "closed.mrmesh",
     "params": {
        "signDetectionMode": "OpenVDB",
        "offset": 5
        }
     },
    {"name": "thicken_unsigned",
     "mesh": "open.mrmesh",
     "params": {
        "signDetectionMode": "Unsigned",
        "offset": 0.5
        }
     },
])
def test_offset_thickening(tmp_path, test_params):
    """
    TODO
    """
    #  Load input meshes
    input_folder = Path(test_files_path) / "algorithms" / "offset"
    case_name = test_params["name"]
    mesh = mlib.loadMesh(input_folder / test_params["mesh"])

    offset_params = mlib.OffsetParameters()
    sign_mode = test_params["params"]["signDetectionMode"]
    offset_params.signDetectionMode = mlib.SignDetectionMode.__members__[sign_mode]
    thicked_mesh = mlib.thickenMesh(mesh=mesh, offset=test_params["params"]["offset"], params=offset_params)

    # # === Verification
    mlib.saveMesh(thicked_mesh, tmp_path / f"{case_name}.mrmesh")
    ref_mesh_path = input_folder / f"{case_name}.mrmesh"
    ref_mesh = mlib.loadMesh(ref_mesh_path)
    # #  check meshes similarity (for extra details on fail)
    with check:
        compare_meshes_similarity(thicked_mesh, ref_mesh)  # diff vs reference usually about 5 verts on 800 overall
    with check:
        self_col_tri = mlib.findSelfCollidingTriangles(thicked_mesh).size()
        assert self_col_tri == 0, f"Mesh should have no self-colliding triangles, actual value is {self_col_tri}"


@pytest.mark.parametrize("test_params", [
    {"name": "double_openMesh_HoleWinding",
     "mesh": "open.mrmesh",
     "params": {
        "signDetectionMode": "HoleWindingRule",
        "offset": 0.5,
        "voxelSize": 0.05
        }
     },
    {"name": "double_plus_OpenVDB",
     "mesh": "morphed.mrmesh",
     "params": {
        "signDetectionMode": "OpenVDB",
        "offset": 2.0,
        "voxelSize": 0.2
        }
     },
    {"name": "double_minus_OpenVDB",
     "mesh": "morphed.mrmesh",
     "params": {
         "signDetectionMode": "OpenVDB",
         "offset": -2.0,
         "voxelSize": 0.2
     }
     },
])
def test_offset_double(tmp_path, test_params):
    """
    TODO
    """
    #  Load input meshes
    input_folder = Path(test_files_path) / "algorithms" / "offset"
    case_name = test_params["name"]
    mesh = mlib.loadMesh(input_folder / test_params["mesh"])

    # params
    offset_params = mlib.OffsetParameters()
    sign_mode = test_params["params"]["signDetectionMode"]
    offset_params.signDetectionMode = mlib.SignDetectionMode.__members__[sign_mode]
    offset_params.voxelSize = test_params["params"]["voxelSize"]

    thicked_mesh = mlib.doubleOffsetMesh(mp=mesh,
                                         offsetA=test_params["params"]["offset"],
                                         offsetB=-test_params["params"]["offset"],
                                         params=offset_params)

    # # === Verification
    mlib.saveMesh(thicked_mesh, tmp_path / f"{case_name}.mrmesh")
    ref_mesh_path = input_folder / f"{case_name}.mrmesh"
    ref_mesh = mlib.loadMesh(ref_mesh_path)
    # #  check meshes similarity (for extra details on fail)
    with check:
        compare_meshes_similarity(thicked_mesh, ref_mesh)  # diff vs reference usually about 5 verts on 800 overall
    with check:
        self_col_tri = mlib.findSelfCollidingTriangles(thicked_mesh).size()
        assert self_col_tri == 0, f"Mesh should have no self-colliding triangles, actual value is {self_col_tri}"


@pytest.mark.parametrize("test_params", [
    {"name": "common_HoleWinding",
     "mesh": "open.mrmesh",
     "params": {
        "signDetectionMode": "HoleWindingRule",
        "offset": 0.5,
        "voxelSize": 0.05,
        }
     },
    {"name": "double_plus_OpenVDB",
     "mesh": "morphed.mrmesh",
     "params": {
        "signDetectionMode": "OpenVDB",
        "offset": 2.0,
        "voxelSize": 0.2
        }
     },
    {"name": "double_minus_OpenVDB",
     "mesh": "morphed.mrmesh",
     "params": {
         "signDetectionMode": "OpenVDB",
         "offset": -2.0,
         "voxelSize": 0.2
     }
     },
])
def test_offset_shell(tmp_path, test_params):
    """
    TODO
    """
    #  Load input meshes
    input_folder = Path(test_files_path) / "algorithms" / "offset"
    case_name = test_params["name"]
    mesh = mlib.loadMesh(input_folder / test_params["mesh"])

    # params
    offset_params = mlib.OffsetParameters()
    sign_mode = test_params["params"]["signDetectionMode"]
    offset_params.signDetectionMode = mlib.SignDetectionMode.__members__[sign_mode]
    offset_params.voxelSize = test_params["params"]["voxelSize"]

    thicked_mesh = mlib.offsetMesh(mp=mesh,
                                         offset=test_params["params"]["offset"],
                                         params=offset_params)

    # # === Verification
    mlib.saveMesh(thicked_mesh, tmp_path / f"{case_name}.mrmesh")
    ref_mesh_path = input_folder / f"{case_name}.mrmesh"
    ref_mesh = mlib.loadMesh(ref_mesh_path)
    # #  check meshes similarity (for extra details on fail)
    with check:
        compare_meshes_similarity(thicked_mesh, ref_mesh)  # diff vs reference usually about 5 verts on 800 overall
    with check:
        self_col_tri = mlib.findSelfCollidingTriangles(thicked_mesh).size()
        assert self_col_tri == 0, f"Mesh should have no self-colliding triangles, actual value is {self_col_tri}"
