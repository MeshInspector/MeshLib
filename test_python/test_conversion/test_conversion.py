from module_helper import *
from pathlib import Path
from constants import test_files_path
from helpers.file_helpers import compare_files


@pytest.mark.parametrize("test_mesh_name", ["fox_geometrik", "Crocodile"])
@pytest.mark.parametrize("ext", ["stl", "obj", "off", "ply", "ctm"])
def test_conversion_from_mrmesh(test_mesh_name, ext, tmp_path):
    input_folder = Path(test_files_path) / "conversion" / "from_mrmesh" / test_mesh_name / ext
    input_mesh = mrmesh.loadMesh(path=str(input_folder / (test_mesh_name + ".mrmesh")))
    filename = test_mesh_name + "." + ext
    with open(tmp_path / filename, "wb") as f:
        mrmesh.saveMesh(mesh=input_mesh, extension="*." + ext, fileHandle=f)

    is_same = compare_files(input_folder / filename, tmp_path / filename)
    assert is_same, f"Converted and reference files are not the same for {filename}"


@pytest.mark.parametrize("test_mesh_name", ["fox_geometrik", "Crocodile"])
@pytest.mark.parametrize("ext", ["stl", "obj", "off", "ply", "ctm"])
def test_conversion_to_mrmesh(test_mesh_name, ext, tmp_path):
    input_folder = Path(test_files_path) / "conversion" / "to_mrmesh" / test_mesh_name / ext
    input_mesh = mrmesh.loadMesh(path=str(input_folder / (test_mesh_name + "." + ext)))
    filename = test_mesh_name + ".mrmesh"
    with open(tmp_path / filename, "wb") as f:
        mrmesh.saveMesh(mesh=input_mesh, extension="*.mrmesh", fileHandle=f)

    is_same = compare_files(input_folder / filename, tmp_path / filename)
    assert is_same, f"Converted and reference files are not the same for {filename} converted from {ext}"
