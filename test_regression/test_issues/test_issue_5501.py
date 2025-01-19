import io
from pathlib import Path

import meshlib.mrmeshpy as mm
import pytest
from constants import test_files_path
from helpers.meshlib_helpers import compare_meshes_similarity


@pytest.mark.smoke
@pytest.mark.bindingsV3
def test_issue_5501_read(tmp_path):
    input_folder = Path(test_files_path) / "issues" / "5501"
    in_file = input_folder / "input.ctm"
    with open(in_file, "rb") as file_handler:
        buf = io.BytesIO(file_handler.read())
        mrmesh = mm.loadMesh(buf, "*.ctm")
        mm.saveMesh(mrmesh, tmp_path / "out_read.ctm")
    compare_meshes_similarity(in_file,  tmp_path / "out_read.ctm")

@pytest.mark.smoke
@pytest.mark.bindingsV3
def test_issue_2899_write(tmp_path):
    input_folder = Path(test_files_path) / "issues" / "5501"
    in_file = input_folder / "input.ctm"
    mrmesh = mm.loadMesh(in_file)
    with io.BytesIO() as byte_stream:
        mm.saveMesh(mrmesh, out=byte_stream, extension="*.ctm")
        with open(tmp_path / "out_write.ctm", "wb") as f:
            f.write(byte_stream.getbuffer())

    compare_meshes_similarity(in_file, tmp_path / "out_write.ctm")
