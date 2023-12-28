import pytest
from module_helper import *
from pathlib import Path
from constants import test_files_path
from helpers.file_helpers import compare_files


@pytest.mark.parametrize("test_mesh_name", ["fox_geometrik", "Crocodile"])
@pytest.mark.parametrize("ext", ["stl", "obj", "off", "ply", "ctm"])
@pytest.mark.parametrize("use_fileHandler", [True, False])
def test_conversion_from_mrmesh(test_mesh_name, ext, use_fileHandler, tmp_path):
    """
    Test conversion from .mrmesh to different formats
    1. Loads mesh from {test_mesh_name}.mrmesh file from test_data/conversion/from_mrmesh/{test_mesh_name}/{ext}
    2. Save it to {test_mesh_name}.{ext} into temporary folder
    3. Compare saved file and file test_data/conversion/from_mrmesh/{test_mesh_name}/{ext}/{test_mesh_name}.{ext}
    If files are same byte-by-byte test passed, otherwise - failed

    use_filehandler: parameter to choose the way of loading and saving mesh. If True - method got fileHandler,
    if False - method got filepath
    """
    # Load mesh
    input_folder = Path(test_files_path) / "conversion" / "from_mrmesh" / test_mesh_name / ext
    if use_fileHandler:
        with open(input_folder / (test_mesh_name + ".mrmesh"), "rb") as mesh_handler:
            input_mesh = mrmesh.loadMesh(fileHandle=mesh_handler, extension="*.mrmesh")
    else:
        input_mesh = mrmesh.loadMesh(path=str(input_folder / (test_mesh_name + ".mrmesh")))
    # Save mesh
    filename = test_mesh_name + "." + ext
    if use_fileHandler:
        with open(tmp_path / filename, "wb") as f:
            mrmesh.saveMesh(mesh=input_mesh, extension="*." + ext, fileHandle=f)
    else:
        mrmesh.saveMesh(mesh=input_mesh, path=str(tmp_path / filename))
    # Compare files
    is_same = compare_files(input_folder / filename, tmp_path / filename)
    assert is_same, f"Converted and reference files are not the same for {filename}"


@pytest.mark.parametrize("test_mesh_name", ["fox_geometrik", "Crocodile"])
@pytest.mark.parametrize("ext", ["stl", "obj", "off", "ply", "ctm"])
@pytest.mark.parametrize("use_fileHandler", [True, False])
def test_conversion_to_mrmesh(test_mesh_name, ext, use_fileHandler, tmp_path):
    """
    Test conversion from different formats to .mrmesh
    1. Loads mesh from {test_mesh_name}.{ext} file from test_data/conversion/to_mrmesh/{test_mesh_name}/{ext}
    2. Save it to {test_mesh_name}.mrmesh into temporary folder
    3. Compare saved file and file test_data/conversion/to_mrmesh/{test_mesh_name}/{ext}/{test_mesh_name}.mrmesh
    If files are same byte-by-byte test passed, otherwise - failed

    use_filehandler: parameter to choose the way of loading and saving mesh. If True - method got fileHandler,
    if False - method got filepath
    """
    # Opening mesh
    input_folder = Path(test_files_path) / "conversion" / "to_mrmesh" / test_mesh_name / ext
    if use_fileHandler:
        with open(input_folder / (test_mesh_name + "." + ext), "rb") as mesh_file:
            input_mesh = mrmesh.loadMesh(fileHandle=mesh_file, extension="*." + ext)
    else:
        input_mesh = mrmesh.loadMesh(path=str(input_folder / (test_mesh_name + "." + ext)))
    # Saving mesh
    filename = test_mesh_name + ".mrmesh"
    if use_fileHandler:
        with open(tmp_path / filename, "wb") as f:
            mrmesh.saveMesh(mesh=input_mesh, extension="*.mrmesh", fileHandle=f)
    else:
        mrmesh.saveMesh(mesh=input_mesh, path=str(tmp_path / filename))

    # Comparing files
    is_same = compare_files(input_folder / filename, tmp_path / filename)
    assert is_same, f"Converted and reference files are not the same for {filename} converted from {ext}"
