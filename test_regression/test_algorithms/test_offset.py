import pytest

from helpers.meshlib_helpers import compare_meshes_similarity
from module_helper import *
from pytest_check import check
from pathlib import Path
from constants import test_files_path


@pytest.mark.parametrize("test_params", [
    pytest.param({"name": "thicken_open_vdb",
                  "mesh": "closed.ctm",
                  "params": {
                      "signDetectionMode": "OpenVDB",
                      "offset": 5
                  }
                  }, id="thicken_open_vdb"),
    pytest.param({"name": "thicken_unsigned",
                  "mesh": "open.ctm",
                  "params": {
                      "signDetectionMode": "Unsigned",
                      "offset": 0.5
                  }
                  }, id="thicken_unsigned"),
    pytest.param({"name": "thicken_self-intersected",
                  "mesh": "self-intersected.ctm",
                  "skip_self-intsc_verif": True,  # Self-intersections presented in input
                  "params": {
                      "signDetectionMode": "WindingRule",
                      "offset": 5
                  }
                  }, id="thicken_self-intersected"),
])
def test_offset_thickening(tmp_path, test_params):
    """
    Tests thickening offset method thickenMesh
    """
    #  Load input meshes
    input_folder = Path(test_files_path) / "algorithms" / "offset"
    case_name = test_params["name"]
    mesh = mrmeshpy.loadMesh(input_folder / test_params["mesh"])

    offset_params = mrmeshpy.GeneralOffsetParameters()
    offset_params.voxelSize = mrmeshpy.suggestVoxelSize(mesh, 5e6)
    sign_mode = test_params["params"]["signDetectionMode"]
    offset_params.signDetectionMode = mrmeshpy.SignDetectionMode.__members__[sign_mode]
    thicked_mesh = mrmeshpy.thickenMesh(mesh, test_params["params"]["offset"], offset_params)

    # === Verification
    mrmeshpy.saveMesh(thicked_mesh, tmp_path / f"{case_name}.ctm")
    ref_mesh_path = input_folder / f"{case_name}.ctm"
    ref_mesh = mrmeshpy.loadMesh(ref_mesh_path)

    with check:
        compare_meshes_similarity(thicked_mesh, ref_mesh)
    with check:
        if "skip_self-intsc_verif" in test_params.keys() and not test_params["skip_self-intsc_verif"]:
            self_col_tri = mrmeshpy.findSelfCollidingTriangles(thicked_mesh).size()
            assert self_col_tri == 0, f"Mesh should have no self-colliding triangles, actual value is {self_col_tri}"
    with check:
        degen_faces = mrmeshpy.findDegenerateFaces(mesh).count()
        assert degen_faces == 0, f"Mesh should have no degenerate faces, actual value is {degen_faces}"


@pytest.mark.parametrize("test_params", [
    pytest.param({"name": "double_plus_OpenVDB",
                  "mesh": "morphed.ctm",
                  "params": {
                      "signDetectionMode": "OpenVDB",
                      "offset": 2.0,
                      "voxelSize": 0.2
                  }
                  }, id="double_plus_OpenVDB"),
    pytest.param({"name": "double_minus_OpenVDB",
                  "mesh": "morphed.ctm",
                  "params": {
                      "signDetectionMode": "OpenVDB",
                      "offset": -2.0,
                      "voxelSize": 0.2
                  }
                  }, id="double_minus_OpenVDB"),
    pytest.param({"name": "double_self-intersected_WindingRule",
                  "mesh": "self-intersected.ctm",
                  "skip_self-intsc_verif": True,  # Self-intersections presented in input
                  "params": {
                      "signDetectionMode": "WindingRule",
                      "offset": -5.0,
                      "voxelSize": 0.5
                  }
                  }, id="double_self-intersected_WindingRule"),
])
def test_offset_double(tmp_path, test_params):
    """
    Tests double offset function (=morphological open/closure) doubleOffsetMesh
    """
    #  Load input meshes
    input_folder = Path(test_files_path) / "algorithms" / "offset"
    case_name = test_params["name"]
    mesh = mrmeshpy.loadMesh(input_folder / test_params["mesh"])

    # params
    offset_params = mrmeshpy.OffsetParameters()
    sign_mode = test_params["params"]["signDetectionMode"]
    offset_params.signDetectionMode = mrmeshpy.SignDetectionMode.__members__[sign_mode]
    offset_params.voxelSize = test_params["params"]["voxelSize"]

    offseted_mesh = mrmeshpy.doubleOffsetMesh(mp=mesh,
                                         offsetA=test_params["params"]["offset"],
                                         offsetB=-test_params["params"]["offset"],
                                         params=offset_params)

    # # === Verification
    mrmeshpy.saveMesh(offseted_mesh, tmp_path / f"{case_name}.ctm")
    ref_mesh_path = input_folder / f"{case_name}.ctm"
    ref_mesh = mrmeshpy.loadMesh(ref_mesh_path)

    with check:
        compare_meshes_similarity(offseted_mesh, ref_mesh, rhsdr_thresh=0.99)
    with check:
        if "skip_self-intsc_verif" in test_params.keys() and not test_params["skip_self-intsc_verif"]:
            self_col_tri = mrmeshpy.findSelfCollidingTriangles(offseted_mesh).size()
            assert self_col_tri == 0, f"Mesh should have no self-colliding triangles, actual value is {self_col_tri}"
    with check:
        degen_faces = mrmeshpy.findDegenerateFaces(mesh).count()
        assert degen_faces == 0, f"Mesh should have no degenerate faces, actual value is {degen_faces}"


@pytest.mark.parametrize("test_params", [
    pytest.param({"name": "shell",
                  "mesh": "morphed.ctm",
                  "params": {
                      "signDetectionMode": "Unsigned",
                      "offset": 1,
                      "voxelSize": 0.2,
                  }
                  }, id="shell"),
])
def test_offset_shell(tmp_path, test_params):
    """
    Tests shell offset and offsetMesh method
    """
    #  Load input meshes
    input_folder = Path(test_files_path) / "algorithms" / "offset"
    case_name = test_params["name"]
    mesh = mrmeshpy.loadMesh(input_folder / test_params["mesh"])

    # params
    offset_params = mrmeshpy.OffsetParameters()
    sign_mode = test_params["params"]["signDetectionMode"]
    offset_params.signDetectionMode = mrmeshpy.SignDetectionMode.__members__[sign_mode]
    offset_params.voxelSize = test_params["params"]["voxelSize"]

    new_mesh = mrmeshpy.offsetMesh(mp=mesh, offset=test_params["params"]["offset"],
                               params=offset_params)

    # === Verification
    mrmeshpy.saveMesh(new_mesh, tmp_path / f"{case_name}.ctm")
    ref_mesh_path = input_folder / f"{case_name}.ctm"
    ref_mesh = mrmeshpy.loadMesh(ref_mesh_path)

    with check:
        compare_meshes_similarity(new_mesh, ref_mesh)
    with check:
        if "skip_self-intsc_verif" in test_params.keys() and not test_params["skip_self-intsc_verif"]:
            self_col_tri = mrmeshpy.findSelfCollidingTriangles(new_mesh).size()
            assert self_col_tri == 0, f"Mesh should have no self-colliding triangles, actual value is {self_col_tri}"
    with check:
        degen_faces = mrmeshpy.findDegenerateFaces(mesh).count()
        assert degen_faces == 0, f"Mesh should have no degenerate faces, actual value is {degen_faces}"


@pytest.mark.parametrize("test_params", [
    pytest.param({"name": "general_sharp_OpenVdb",
                  "mesh": "morphed.ctm",
                  "skip_self-intsc_verif": True,  # sharpening produces intersections, that decided not to fix for now
                  "params": {
                      "signDetectionMode": "OpenVDB",
                      "offset": 1,
                      "voxelSize": 0.1,
                      "mode": "Sharpening"
                  }
                  }, id="general_sharp_OpenVdb"),
    pytest.param({"name": "general_smooth_OpenVdb",
                  "mesh": "morphed.ctm",
                  "params": {
                      "signDetectionMode": "OpenVDB",
                      "offset": 1,
                      "voxelSize": 0.2,
                      "mode": "Smooth"
                  }
                  }, id="general_smooth_OpenVdb", marks=pytest.mark.smoke),
    pytest.param({"name": "general_Standard_ProjectionNormal",
                  "mesh": "morphed.ctm",
                  "params": {
                      "signDetectionMode": "ProjectionNormal",
                      "offset": 1,
                      "voxelSize": 0.2,
                      "mode": "Standard"
                  }
                  }, id="general_Standard_ProjectionNormal"),
    pytest.param({"name": "general_Standard_WindingRule",
                  "mesh": "morphed.ctm",
                  "skip_self-intsc_verif": True,  # sharpening produces intersections, that decided not to fix for now
                  "params": {
                      "signDetectionMode": "WindingRule",
                      "offset": 1,
                      "voxelSize": 0.2,
                      "mode": "Sharpening"
                  }
                  }, id="general_Standard_WindingRule"),
    pytest.param({"name": "general_Standard_open_HoleWindingRule",
                  "mesh": "open.ctm",
                  "params": {
                      "signDetectionMode": "HoleWindingRule",
                      "offset": 1,
                      "voxelSize": 0.2,
                      "mode": "Standard"
                  }
                  }, id="general_Standard_open_HoleWindingRule"),
    pytest.param({"name": "general_Standard_self-intersected_HoleWindingRule",
                  "mesh": "self-intersected.ctm",
                  "skip_self-intsc_verif": True,
                  "params": {
                      "signDetectionMode": "WindingRule",
                      "offset": 5,
                      "voxelSize": 1,
                      "mode": "Standard"
                  }
                  }, id="general_Standard_self-intersected_HoleWindingRule")
])
def test_offset_general(tmp_path, test_params):
    """
    Tests generalOffsetMesh with combinations of GeneralOffsetParametersMode (Sharpening, Smooth, Standard)
    """
    #  Load input meshes
    input_folder = Path(test_files_path) / "algorithms" / "offset"
    case_name = test_params["name"]
    mesh = mrmeshpy.loadMesh(input_folder / test_params["mesh"])

    # params
    offset_params = mrmeshpy.GeneralOffsetParameters()
    offset_params.signDetectionMode = mrmeshpy.SignDetectionMode.__members__[test_params["params"]["signDetectionMode"]]
    offset_params.mode = mrmeshpy.GeneralOffsetParametersMode.__members__[test_params["params"]["mode"]]
    offset_params.voxelSize = test_params["params"]["voxelSize"]

    new_mesh = mrmeshpy.generalOffsetMesh(mp=mesh, offset=test_params["params"]["offset"],
                                      params=offset_params)

    # === Verification
    mrmeshpy.saveMesh(new_mesh, tmp_path / f"{case_name}.ctm")
    ref_mesh_path = input_folder / f"{case_name}.ctm"
    ref_mesh = mrmeshpy.loadMesh(ref_mesh_path)

    with check:
        compare_meshes_similarity(new_mesh, ref_mesh)

    with check:
        if "skip_self-intsc_verif" in test_params.keys() and not test_params["skip_self-intsc_verif"]:
            self_col_tri = mrmeshpy.findSelfCollidingTriangles(new_mesh).size()
            assert self_col_tri == 0, f"Mesh should have no self-colliding triangles, actual value is {self_col_tri}"
    with check:
        degen_faces = mrmeshpy.findDegenerateFaces(mesh).count()
        assert degen_faces == 0, f"Mesh should have no degenerate faces, actual value is {degen_faces}"


@pytest.mark.bindingsV3
def test_offset_weighted_shell(tmp_path):
    """
    Tests weightedMeshShell method with vertex-based weights
    """
    # Load input mesh
    input_folder = Path(test_files_path) / "algorithms" / "offset"
    case_name = "weighted_offset"
    mesh = mrmeshpy.loadMesh(input_folder / "Torus_default.ctm")

    # Create weights for vertices
    vertex_count = mesh.points.size()
    scalars = mrmeshpy.VertScalars(vertex_count)
    for i in range(vertex_count):
        weight = abs(mesh.points.vec[i].x / 5) + 0.1
        scalars.vec[i] = weight

    # Setup parameters
    params = mrmeshpy.WeightedShell.ParametersMetric()
    params.offset = 0.05
    params.voxelSize = 0.1
    params.dist.maxWeight = max(scalars.vec)

    # Run weightedMeshShell
    new_mesh = mrmeshpy.WeightedShell.meshShell(mesh, scalars, params)

    # === Verification
    mrmeshpy.saveMesh(new_mesh, tmp_path / f"{case_name}.ctm")
    ref_mesh_path = input_folder / f"{case_name}.ctm"
    ref_mesh = mrmeshpy.loadMesh(ref_mesh_path)

    with check:
        compare_meshes_similarity(new_mesh, ref_mesh)
    with check:
        self_col_tri = mrmeshpy.findSelfCollidingTriangles(new_mesh).size()
        assert self_col_tri == 0, f"Mesh should have no self-colliding triangles, actual value is {self_col_tri}"
    with check:
        degen_faces = mrmeshpy.findDegenerateFaces(mesh).count()
        assert degen_faces == 0, f"Mesh should have no degenerate faces, actual value is {degen_faces}"
