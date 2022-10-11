from helper import *
import pytest


def test_find_max_mesh_distance_sq():
    torus = mrmesh.makeTorus(2, 1, 10, 10, None)
    torus2 = mrmesh.makeTorus(2, 1, 10, 10, None)

    transVector = mrmesh.Vector3f()
    transVector.x = 40
    transVector.y = 40
    transVector.z = 40
    diffXf = mrmesh.AffineXf3f.translation(transVector)
    torus2.transform(diffXf)

    distSq = mrmesh.findMaxDistanceSq(
        mrmesh.MeshPart(torus),
        mrmesh.MeshPart(torus2),
        diffXf.inverse(),
        1e35)

    assert (distSq < 69.28205**2)
