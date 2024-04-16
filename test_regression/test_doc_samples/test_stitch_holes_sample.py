import pytest

from constants import test_files_path, DEFAULT_RHAUSDORF_THRESHOLD
from helpers.meshlib_helpers import relative_hausdorff
from module_helper import *
from pathlib import Path


@pytest.mark.smoke
def test_stitch_holes_sample(tmp_path):
    input_folder = Path(test_files_path) / "doc_samples" / "stitch_holes"

    # === Sample code
    import meshlib.mrmeshpy as mrmeshpy

    # Load meshes
    meshA = mrmeshpy.loadMesh(str(input_folder / "crocodile_sad.stl"))
    meshB = mrmeshpy.loadMesh(str(input_folder / "crocodile_tail.stl"))

    # Unite meshes
    mesh = mrmeshpy.mergeMeshes([meshA, meshB])

    # Find holes
    edges = mesh.topology.findHoleRepresentiveEdges()

    # Connect two holes
    params = mrmeshpy.StitchHolesParams()
    params.metric = mrmeshpy.getUniversalMetric(mesh)
    mrmeshpy.buildCylinderBetweenTwoHoles(mesh, edges[0], edges[1], params)

    # Save result
    mrmeshpy.saveMesh(mesh, str(tmp_path / "stitchedMesh.stl"))

    # Verification
    assert relative_hausdorff(tmp_path / "stitchedMesh.stl",
                              input_folder / "stitchedMesh.stl") > DEFAULT_RHAUSDORF_THRESHOLD
