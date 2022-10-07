from helper import *
import pytest


def test_fix_undercuts():
    torus = mrmesh.makeTorusWithUndercut(2, 1, 1.5, 10, 10, None)

    dir = mrmesh.Vector3f()
    dir.x = 0
    dir.y = 0
    dir.z = 1

    mrmesh.fixUndercuts(torus, dir, 0.2, 0.)

    assert (torus.points.vec.size() > 2900)
