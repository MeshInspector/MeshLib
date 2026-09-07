from pathlib import Path

import pytest
from module_helper import *
from constants import test_files_path


def _collect_meshes(obj, out):
    if isinstance(obj, mrmeshpy.ObjectMesh) and obj.mesh() is not None:
        out.append(obj)
    for child in obj.children():
        _collect_meshes(child, out)


@pytest.mark.smoke
def test_step_multibody_split():
    """
    PR #6240: a STEP part containing several solids must load as separate
    ObjectMesh objects instead of one merged mesh.

    The fixture is a single part holding two tetrahedra placed apart along X
    (one near the origin, one near x=100).
    """
    input_file = Path(test_files_path) / "conversion" / "step_multibody" / "two_bodies.step"
    scene = mrmeshpy.loadSceneObject(input_file)

    meshes = []
    _collect_meshes(scene, meshes)

    # the two solids load as two separate ObjectMesh objects
    assert len(meshes) == 2

    # geometry is preserved, nothing lost or duplicated: two tetrahedra, 4 triangles each
    assert sum(m.mesh().topology.numValidFaces() for m in meshes) == 8

    # placement is preserved: the bodies are not collapsed onto each other
    max_x = max(m.getWorldBox().max.x for m in meshes)
    assert max_x == pytest.approx(110.0, abs=1.0)
