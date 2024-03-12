import pytest
from module_helper import *
from pathlib import Path
from constants import test_files_path
from helpers.file_helpers import compare_file_with_multiple_references, get_reference_files_list


@pytest.mark.parametrize("test_points_name", ["bicao"])
@pytest.mark.parametrize("ext", ["asc", "ctm"])
@pytest.mark.parametrize("use_fileHandler", [True, False])
def test_conversion_from_ply(test_points_name, ext, use_fileHandler, tmp_path):
    """
    Test conversion from cloud points .ply to different formats
    1. Loads cload points from {test_points_name}.ply file from
    test_data/conversion/points_from_ply/{test_mesh_name}/{ext}
    2. Save it to {test_points_name}.{ext} into temporary folder
    3. Compare saved file and file test_data/conversion/points_from_ply/{test_points_name}/{ext}/{test_points_name}.{ext}
    If files are same byte-by-byte test passed, otherwise - failed

    use_filehandler: parameter to choose the way of loading and saving points file. If True - method got fileHandler,
    if False - method got filepath
    """
    # Load points
    input_folder = Path(test_files_path) / "conversion" / "points_from_ply" / test_points_name / ext
    if use_fileHandler:
        with open(input_folder / (test_points_name + ".ply"), "rb") as file_handler:
            input_mesh = mrmesh.loadPoints(fileHandle=file_handler, extension="*.ply")
    else:
        input_mesh = mrmesh.loadPoints(path=str(input_folder / (test_points_name + ".ply")))
    # Save points
    filename = test_points_name + "." + ext
    if use_fileHandler:
        with open(tmp_path / filename, "wb") as f:
            mrmesh.savePoints(pointCloud=input_mesh, extension="*." + ext, fileHandle=f)
    else:
        mrmesh.savePoints(pointCloud=input_mesh, path=str(tmp_path / filename))
    # Comparing files
    ref_files_list = get_reference_files_list(input_folder / filename)
    is_same_found = compare_file_with_multiple_references(tmp_path / filename, ref_files_list)
    assert is_same_found, f"Converted file doesn't match to any reference"


@pytest.mark.parametrize("test_points_name", ["bicao"])
@pytest.mark.parametrize("ext", ["asc", "ctm", "xyz"])
@pytest.mark.parametrize("use_fileHandler", [True, False])
def test_conversion_to_ply(test_points_name, ext, use_fileHandler, tmp_path):
    """
    Test conversion from different points formats to .ply
    1. Loads cloud points from {test_points_name}.{ext} file from
    test_data/conversion/points_to_ply/{test_mesh_name}/{ext}
    2. Save it to {test_points_name}.ply into temporary folder
    3. Compare saved file and file test_data/conversion/points_to_ply/{test_points_name}/{ext}/{test_points_name}.ply
    If files are same byte-by-byte test passed, otherwise - failed

    use_filehandler: parameter to choose the way of loading and saving points file. If True - method got fileHandler,
    if False - method got filepath
    """
    # Opening mesh
    input_folder = Path(test_files_path) / "conversion" / "points_to_ply" / test_points_name / ext
    if use_fileHandler:
        with open(input_folder / (test_points_name + "." + ext), "rb") as file_handler:
            input_points = mrmesh.loadPoints(fileHandle=file_handler, extension="*." + ext)
    else:
        input_points = mrmesh.loadPoints(path=str(input_folder / (test_points_name + "." + ext)))
    # Saving mesh
    filename = test_points_name + ".ply"
    if use_fileHandler:
        with open(tmp_path / filename, "wb") as f:
            mrmesh.savePoints(pointCloud=input_points, extension="*.ply", fileHandle=f)
    else:
        mrmesh.savePoints(pointCloud=input_points, path=str(tmp_path / filename))

    # Comparing files

    ref_files_list = get_reference_files_list(input_folder / filename)
    is_same_found = compare_file_with_multiple_references(tmp_path / filename, ref_files_list)

    assert is_same_found, f"Converted and reference files are not the same for {filename} converted from {ext}"
