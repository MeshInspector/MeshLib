from pathlib import Path

import pytest
from module_helper import *
from constants import test_files_path


def _collect_meshes(obj, out):
    if isinstance(obj, mrmeshpy.ObjectMesh) and obj.meshPtr() is not None:
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
    assert sum(m.meshPtr().topology.numValidFaces() for m in meshes) == 8

    # placement is preserved: the bodies are not collapsed onto each other
    max_x = max(m.getWorldBox().max.x for m in meshes)
    assert max_x == pytest.approx(110.0, abs=1.0)


@pytest.mark.smoke
def test_step_force_load_sub_shapes():
    """
    A single-body STEP part loads as one ObjectMesh; with
    StepLoadSettings.forceLoadSubShapes it is split into per-body children instead.

    The fixture is a single part holding one tetrahedron placed near x=100.
    """
    input_file = Path(test_files_path) / "conversion" / "step_multibody" / "one_body.step"

    # default: no extra group level, the part itself is the mesh object
    scene = mrmeshpy.MeshLoad.fromSceneStepFile(input_file)
    assert len(scene.children()) == 1
    part = scene.children()[0]
    assert isinstance(part, mrmeshpy.ObjectMesh)
    assert part.meshPtr().topology.numValidFaces() == 4
    assert len(part.children()) == 0

    step_settings = mrmeshpy.MeshLoad.StepLoadSettings()
    step_settings.forceLoadSubShapes = True
    scene = mrmeshpy.MeshLoad.fromSceneStepFile(input_file, stepSettings=step_settings)
    assert len(scene.children()) == 1
    group = scene.children()[0]
    assert not isinstance(group, mrmeshpy.ObjectMesh)
    assert len(group.children()) == 1

    # only the scene structure changes: same geometry, same placement
    body = group.children()[0]
    assert isinstance(body, mrmeshpy.ObjectMesh)
    assert body.meshPtr().topology.numValidFaces() == 4
    assert body.getWorldBox().max.x == pytest.approx(110.0, abs=1e-3)
