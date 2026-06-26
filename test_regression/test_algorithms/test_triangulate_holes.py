from module_helper import *
from pathlib import Path
from pytest_check import check
from constants import test_files_path

import pytest


@pytest.mark.smoke
def test_triangulate_holes():
    """
    Test triangulateHoles algorithm.
    The input is a sphere split into 4 separate open pieces (4 holes, 4 components).
    triangulateHoles must fill all the holes, producing a closed mesh
    consisting of a single connected component.
    """
    # Load input mesh
    input_folder = Path(test_files_path) / "algorithms" / "triangulate_holes"
    mesh = mrmeshpy.loadMesh(input_folder / "Sphere4pieces.ctm")

    # Fill all the holes of the mesh
    res = mrmeshpy.triangulateHoles(mesh)

    # === Verification
    with check:
        assert res, "triangulateHoles should succeed (return True)"
    with check:
        num_holes = mesh.topology.findNumHoles()
        assert num_holes == 0, f"Resulting mesh must be closed (no holes), actual number of holes is {num_holes}"
    with check:
        num_components = mrmeshpy.MeshComponents.getNumComponents(mesh)
        assert num_components == 1, f"Resulting mesh must have a single connected component, actual value is {num_components}"
