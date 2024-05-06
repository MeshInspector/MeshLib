import pytest
from helper import *


def test_collision():
    torus = mrmesh.makeTorus(2, 1, 10, 10, None)
    torus2 = mrmesh.makeTorus(2, 1, 10, 10, None)

    transVector = mrmesh.Vector3f()
    transVector.x = 0.5
    transVector.y = 1
    transVector.z = 1
    diffXf = mrmesh.AffineXf3f.translation(transVector)
    torus2.transform(diffXf)

    xf = mrmesh.AffineXf3f()
    torus1 = torus
    pairs = mrmesh.findCollidingTriangles(torus1, torus2)
    # at least 100 triangles should collide for that transforms
    assert len(pairs) > 103
