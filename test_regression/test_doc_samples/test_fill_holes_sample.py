from constants import test_files_path, DEFAULT_RHAUSDORF_THRESHOLD
from helpers.meshlib_helpers import relative_hausdorff
from module_helper import *
from pathlib import Path


def test_fill_holes_sample(tmp_path):
    input_folder = Path(test_files_path) / "doc_samples" / "fill_holes"

    # === Sample code

    import meshlib.mrmeshpy as mrmeshpy

    # Load mesh
    mesh = mrmeshpy.loadMesh(str(input_folder / "detail_with_holes.ctm"))

    # Find single edge for each hole in mesh
    holeEdges = mesh.topology.findHoleRepresentiveEdges()

    for e in holeEdges:
        # Setup filling parameters
        params = mrmeshpy.FillHoleParams()
        params.metric = mrmeshpy.getUniversalMetric(mesh)
        # Fill hole represented by `e`
        mrmeshpy.fillHole(mesh, e, params)

    # Save result
    mrmeshpy.saveMesh(mesh, str(tmp_path / "filledMesh.ctm"))
    #  === Verification
    assert relative_hausdorff(tmp_path / "filledMesh.ctm",
                              input_folder / "filledMesh.ctm") > DEFAULT_RHAUSDORF_THRESHOLD
