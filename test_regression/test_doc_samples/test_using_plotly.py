from module_helper import *
from pathlib import Path
from constants import test_files_path


def test_using_plotly(tmp_path):
    input_folder = Path(test_files_path) / "doc_samples" / "plotly"

    # === Sample code
    from meshlib import mrmeshpy as mm
    from meshlib import mrmeshnumpy as mn
    import numpy as np
    import plotly.graph_objects as go

    # load mesh
    mesh = mm.loadMesh(mm.Path(str(Path(input_folder / "fox_geometrik.stl"))))
    # extract numpy arrays
    verts = mn.getNumpyVerts(mesh)
    faces = mn.getNumpyFaces(mesh.topology)
    # print(verts)

    # prepare data for plotly
    vertsT = np.transpose(verts)
    facesT = np.transpose(faces)

    # draw
    fig = go.Figure(data=[
        go.Mesh3d(
            x=vertsT[0],
            y=vertsT[1],
            z=vertsT[2],
            i=facesT[0],
            j=facesT[1],
            k=facesT[2]
        )
    ])

    #  === Verification
    assert isinstance(verts, np.ndarray)
    assert isinstance(faces, np.ndarray)
