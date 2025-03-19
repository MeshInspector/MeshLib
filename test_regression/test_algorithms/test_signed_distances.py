from module_helper import *
from pathlib import Path
from pytest_check import check
from constants import test_files_path
from helpers.meshlib_helpers import compare_meshes_similarity, compare_mesh

import pytest

def test_signed_distance():
    input_folder = Path(test_files_path) / "algorithms" / "signed_distance"
    mesh1 = mrmeshpy.loadMesh(input_folder / "beethoven.ctm")
    mesh2 = mrmeshpy.loadMesh(input_folder / "Torus.ctm")
    a = mrmeshpy.findSignedDistance(mesh1, mesh2)
    print(a.signedDist)
    assert a.signedDist == pytest.approx(-1.8215560)


def test_signed_distance_point():
    input_folder = Path(test_files_path) / "algorithms" / "signed_distance"
    mesh1 = mrmeshpy.loadMesh(input_folder / "beethoven.ctm")
    point = mrmeshpy.Vector3f(1, 2, 3)
    a = mrmeshpy.findSignedDistance(point, mesh1)
    print(a.dist)
    assert a.dist == pytest.approx(-2.8215560)

@pytest.mark.bindingsV3
def test_signed_distances_meshes():
    input_folder = Path(test_files_path) / "algorithms" / "signed_distance"
    mesh1 = mrmeshpy.loadMesh(input_folder / "beethoven.ctm")
    mesh2 = mrmeshpy.loadMesh(input_folder / "Torus.ctm")
    a = mrmeshpy.findSignedDistances(mesh1, mesh2)

    assert a.data() == pytest.approx(0.17844390869140625)
    assert a.front() == pytest.approx(0.17844390869140625)
    assert a.back() == pytest.approx(0.5686243772506714)
