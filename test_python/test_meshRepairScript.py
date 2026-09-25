import os
import sys

import pytest
from helper import *

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python_scripts"))
import mesh_repair


def test_mesh_repair_script():
    # two intersecting spheres as separate components, with a small hole in one of them
    mesh = mrmesh.makeSphere(mrmesh.SphereParams())
    other = mrmesh.makeSphere(mrmesh.SphereParams())
    other.transform(mrmesh.AffineXf3f.translation(mrmesh.Vector3f(1, 0, 0)))
    mesh.addMesh(other)
    hole = mrmesh.FaceBitSet(mesh.topology.faceSize())
    hole.set(mrmesh.FaceId(0))
    mesh.deleteFaces(hole)

    diagonal = mesh.computeBoundingBox().diagonal()
    tolerance = diagonal * 2.5e-3
    assert mesh_repair.has_issues(mesh, 0.002, diagonal / 3.0, tolerance)

    mesh = mesh_repair.iterative_repair(mesh, tolerance)
    assert not mesh_repair.has_issues(mesh, 0.002, diagonal / 3.0, tolerance)
    assert mesh.topology.findNumHoles() == 0
