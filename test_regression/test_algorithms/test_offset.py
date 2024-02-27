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
    {"name": "thicken_self-intersected",
     "mesh": "self-intersected.mrmesh",
     "skip_self-intsc_verif": True,  # Self-intersections presented in input
     "params": {
         "signDetectionMode": "WindingRule",
         "offset": 5
     }
     },
])
def test_offset_thickening(tmp_path, test_params):
    """
    Tests thickening offset method thickenMesh
    """
    #  Load input meshes
    input_folder = Path(test_files_path) / "algorithms" / "offset"
    case_name = test_params["name"]
    mesh = mlib.loadMesh(input_folder / test_params["mesh"])

    offset_params = mlib.OffsetParameters()
    sign_mode = test_params["params"]["signDetectionMode"]
    offset_params.signDetectionMode = mlib.SignDetectionMode.__members__[sign_mode]
    thicked_mesh = mlib.thickenMesh(mesh=mesh, offset=test_params["params"]["offset"], params=offset_params)

    # === Verification
    mlib.saveMesh(thicked_mesh, tmp_path / f"{case_name}.mrmesh")
    ref_mesh_path = input_folder / f"{case_name}.mrmesh"
    ref_mesh = mlib.loadMesh(ref_mesh_path)

    with check:
        compare_meshes_similarity(thicked_mesh, ref_mesh)
    with check:
        if "skip_self-intsc_verif" in test_params.keys() and not test_params["skip_self-intsc_verif"]:
            self_col_tri = mlib.findSelfCollidingTriangles(thicked_mesh).size()
            assert self_col_tri == 0, f"Mesh should have no self-colliding triangles, actual value is {self_col_tri}"
    with check:
        degen_faces = mlib.findDegenerateFaces(mesh).count()
        assert degen_faces == 0, f"Mesh should have no degenerate faces, actual value is {degen_faces}"


@pytest.mark.parametrize("test_params", [
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
    {"name": "double_self-intersected_WindingRule",
     "mesh": "self-intersected.mrmesh",
     "skip_self-intsc_verif": True,  # Self-intersections presented in input
     "params": {
         "signDetectionMode": "WindingRule",
         "offset": -5.0,
         "voxelSize": 0.5
        }
     },
])
def test_offset_double(tmp_path, test_params):
    """
    Tests double offset function (=morphological open/closure) doubleOffsetMesh
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

    offseted_mesh = mlib.doubleOffsetMesh(mp=mesh,
                                         offsetA=test_params["params"]["offset"],
                                         offsetB=-test_params["params"]["offset"],
                                         params=offset_params)

    # # === Verification
    mlib.saveMesh(offseted_mesh, tmp_path / f"{case_name}.mrmesh")
    ref_mesh_path = input_folder / f"{case_name}.mrmesh"
    ref_mesh = mlib.loadMesh(ref_mesh_path)

    with check:
        compare_meshes_similarity(offseted_mesh, ref_mesh, rhsdr_thresh=0.99)
    with check:
        if "skip_self-intsc_verif" in test_params.keys() and not test_params["skip_self-intsc_verif"]:
            self_col_tri = mlib.findSelfCollidingTriangles(offseted_mesh).size()
            assert self_col_tri == 0, f"Mesh should have no self-colliding triangles, actual value is {self_col_tri}"
    with check:
        degen_faces = mlib.findDegenerateFaces(mesh).count()
        assert degen_faces == 0, f"Mesh should have no degenerate faces, actual value is {degen_faces}"


@pytest.mark.parametrize("test_params", [
    {"name": "shell",
     "mesh": "morphed.mrmesh",
     "params": {
         "signDetectionMode": "Unsigned",
         "offset": 1,
         "voxelSize": 0.2,
     }
     },
])
def test_offset_shell(tmp_path, test_params):
    """
    Tests shell offset and offsetMesh method
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

    new_mesh = mlib.offsetMesh(mp=mesh, offset=test_params["params"]["offset"],
                               params=offset_params)

    # === Verification
    mlib.saveMesh(new_mesh, tmp_path / f"{case_name}.mrmesh")
    ref_mesh_path = input_folder / f"{case_name}.mrmesh"
    ref_mesh = mlib.loadMesh(ref_mesh_path)

    with check:
        compare_meshes_similarity(new_mesh, ref_mesh)
    with check:
        if "skip_self-intsc_verif" in test_params.keys() and not test_params["skip_self-intsc_verif"]:
            self_col_tri = mlib.findSelfCollidingTriangles(new_mesh).size()
            assert self_col_tri == 0, f"Mesh should have no self-colliding triangles, actual value is {self_col_tri}"
    with check:
        degen_faces = mlib.findDegenerateFaces(mesh).count()
        assert degen_faces == 0, f"Mesh should have no degenerate faces, actual value is {degen_faces}"


@pytest.mark.parametrize("test_params", [
    {"name": "general_sharp_OpenVdb",
     "mesh": "morphed.mrmesh",
     "skip_self-intsc_verif": True,  # sharpening produces intersections, that decided not to fix for now
     "params": {
         "signDetectionMode": "OpenVDB",
         "offset": 1,
         "voxelSize": 0.1,
         "mode": "Sharpening"
     }
     },
    {"name": "general_smooth_OpenVdb",
     "mesh": "morphed.mrmesh",
     "params": {
         "signDetectionMode": "OpenVDB",
         "offset": 1,
         "voxelSize": 0.2,
         "mode": "Smooth"
     }
     },
    {"name": "general_Standard_ProjectionNormal",
     "mesh": "morphed.mrmesh",
     "params": {
         "signDetectionMode": "ProjectionNormal",
         "offset": 1,
         "voxelSize": 0.2,
         "mode": "Standard"
     }
     },
    {"name": "general_Standard_WindingRule",
     "mesh": "morphed.mrmesh",
     "skip_self-intsc_verif": True,  # sharpening produces intersections, that decided not to fix for now
     "params": {
         "signDetectionMode": "WindingRule",
         "offset": 1,
         "voxelSize": 0.2,
         "mode": "Sharpening"
     }
     },
    {"name": "general_Standard_open_HoleWindingRule",
     "mesh": "open.mrmesh",
     "params": {
         "signDetectionMode": "HoleWindingRule",
         "offset": 1,
         "voxelSize": 0.2,
         "mode": "Standard"
     }
     },
    {"name": "general_Standard_self-intersected_HoleWindingRule",
     "mesh": "self-intersected.mrmesh",
     "skip_self-intsc_verif": True,
     "params": {
         "signDetectionMode": "WindingRule",
         "offset": 5,
         "voxelSize": 1,
         "mode": "Standard"
     }
     }
])
def test_offset_general(tmp_path, test_params):
    """
    Tests generalOffsetMesh with combinations of GeneralOffsetParametersMode (Sharpening, Smooth, Standard)
    """
    #  Load input meshes
    input_folder = Path(test_files_path) / "algorithms" / "offset"
    case_name = test_params["name"]
    mesh = mlib.loadMesh(input_folder / test_params["mesh"])

    # params
    offset_params = mlib.GeneralOffsetParameters()
    offset_params.signDetectionMode = mlib.SignDetectionMode.__members__[test_params["params"]["signDetectionMode"]]
    offset_params.mode = mlib.GeneralOffsetParametersMode.__members__[test_params["params"]["mode"]]
    offset_params.voxelSize = test_params["params"]["voxelSize"]

    new_mesh = mlib.generalOffsetMesh(mp=mesh, offset=test_params["params"]["offset"],
                                      params=offset_params)

    # === Verification
    mlib.saveMesh(new_mesh, tmp_path / f"{case_name}.mrmesh")
    ref_mesh_path = input_folder / f"{case_name}.mrmesh"
    ref_mesh = mlib.loadMesh(ref_mesh_path)

    with check:
        compare_meshes_similarity(new_mesh, ref_mesh)

    with check:
        if "skip_self-intsc_verif" in test_params.keys() and not test_params["skip_self-intsc_verif"]:
            self_col_tri = mlib.findSelfCollidingTriangles(new_mesh).size()
            assert self_col_tri == 0, f"Mesh should have no self-colliding triangles, actual value is {self_col_tri}"
    with check:
        degen_faces = mlib.findDegenerateFaces(mesh).count()
        assert degen_faces == 0, f"Mesh should have no degenerate faces, actual value is {degen_faces}"