import pytest
from module_helper import *
from constants import test_files_path
from pathlib import Path
from helpers.file_helpers import compare_file_with_multiple_references, get_reference_files_list


@pytest.mark.smoke
@pytest.mark.parametrize("test_mesh_name", ["fox_geometrik"])
@pytest.mark.parametrize("ext", ["stl", "obj", "off", "ply", "ctm"])
@pytest.mark.parametrize("use_fileHandler", [True, False])
def test_conversion_from_mrmesh(test_mesh_name, ext, use_fileHandler, tmp_path):
    """
    Test conversion from .mrmesh to different formats
    1. Loads mesh from {test_mesh_name}.mrmesh file from test_data/conversion/meshes_from_mrmesh/{test_mesh_name}/{ext}
    2. Save it to {test_mesh_name}.{ext} into temporary folder
    3. Compare saved file and file test_data/conversion/meshes_from_mrmesh/{test_mesh_name}/{ext}/{test_mesh_name}.{ext}
    If files are same byte-by-byte test passed, otherwise - failed

    use_filehandler: parameter to choose the way of loading and saving mesh. If True - method got fileHandler,
    if False - method got filepath
    """
    # Load mesh
    input_folder = Path(test_files_path) / "conversion" / "meshes_from_mrmesh" / test_mesh_name / ext
    if use_fileHandler:
        with open(input_folder / (test_mesh_name + ".mrmesh"), "rb") as mesh_handler:
            input_mesh = mrmeshpy.loadMesh(mesh_handler, "*.mrmesh")
    else:
        input_mesh = mrmeshpy.loadMesh(str(input_folder / (test_mesh_name + ".mrmesh")))
    # Save mesh
    filename = test_mesh_name + "." + ext
    if use_fileHandler:
        with open(tmp_path / filename, "wb") as f:
            mrmeshpy.saveMesh(input_mesh, "*." + ext, f)
    else:
        mrmeshpy.saveMesh(input_mesh, str(tmp_path / filename))
    # Comparing files
    ref_files_list = get_reference_files_list(input_folder / filename)
    is_same_found = compare_file_with_multiple_references(tmp_path / filename, ref_files_list)
    assert is_same_found, f"Converted file doesn't match to any reference"


@pytest.mark.smoke
@pytest.mark.parametrize("test_mesh_name", ["fox_geometrik"])
@pytest.mark.parametrize("ext", ["stl", "obj", "off", "ply", "ctm"])
@pytest.mark.parametrize("use_fileHandler", [True, False])
def test_conversion_to_mrmesh(test_mesh_name, ext, use_fileHandler, tmp_path):
    """
    Test conversion from different formats to .mrmesh
    1. Loads mesh from {test_mesh_name}.{ext} file from test_data/conversion/meshes_to_mrmesh/{test_mesh_name}/{ext}
    2. Save it to {test_mesh_name}.mrmesh into temporary folder
    3. Compare saved file and file test_data/conversion/meshes_to_mrmesh/{test_mesh_name}/{ext}/{test_mesh_name}.mrmesh
    If files are same byte-by-byte test passed, otherwise - failed

    use_filehandler: parameter to choose the way of loading and saving mesh. If True - method got fileHandler,
    if False - method got filepath
    """
    # Opening mesh
    input_folder = Path(test_files_path) / "conversion" / "meshes_to_mrmesh" / test_mesh_name / ext
    if use_fileHandler:
        with open(input_folder / (test_mesh_name + "." + ext), "rb") as mesh_file:
            input_mesh = mrmeshpy.loadMesh(mesh_file, "*." + ext)
    else:
        input_mesh = mrmeshpy.loadMesh(str(input_folder / (test_mesh_name + "." + ext)))
    # Saving mesh
    filename = test_mesh_name + ".mrmesh"
    if use_fileHandler:
        with open(tmp_path / filename, "wb") as f:
            mrmeshpy.saveMesh(input_mesh, "*.mrmesh", f)
    else:
        mrmeshpy.saveMesh(input_mesh, str(tmp_path / filename))

    # Comparing files

    ref_files_list = get_reference_files_list(input_folder / filename)
    is_same_found = compare_file_with_multiple_references(tmp_path / filename, ref_files_list)

    assert is_same_found, f"Converted and reference files are not the same for {filename} converted from {ext}"
