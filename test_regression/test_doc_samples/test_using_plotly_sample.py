import pytest

from module_helper import *
from pathlib import Path
from constants import test_files_path


@pytest.mark.smoke
def test_using_plotly_sample(tmp_path):
    input_folder = Path(test_files_path) / "doc_samples" / "plotly"

    # === Sample code
    import numpy as np

    # load mesh
    mesh = mrmeshpy.loadMesh(str(Path(input_folder / "fox_geometrik.stl")))
    # extract numpy arrays
    verts = mrmeshnumpy.getNumpyVerts(mesh)
    faces = mrmeshnumpy.getNumpyFaces(mesh.topology)

    #  === Verification
    assert isinstance(verts, np.ndarray)
    assert isinstance(faces, np.ndarray)
