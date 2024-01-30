from module_helper import *
from pathlib import Path
from constants import test_files_path


def test_using_plotly(tmp_path):
    input_folder = Path(test_files_path) / "doc_samples" / "plotly"

    # === Sample code
    from meshlib import mrmeshpy as mm
    from meshlib import mrmeshnumpy as mn
    import numpy as np

    # load mesh
    mesh = mm.loadMesh(str(Path(input_folder / "fox_geometrik.stl")))
    # extract numpy arrays
    verts = mn.getNumpyVerts(mesh)
    faces = mn.getNumpyFaces(mesh.topology)

    #  === Verification
    assert isinstance(verts, np.ndarray)
    assert isinstance(faces, np.ndarray)
