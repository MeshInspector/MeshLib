import pytest
from helper import *


def test_fix_undercuts():
    torus = mrmesh.makeTorusWithUndercut(2, 1, 1.5, 10, 10, None)

    dir = mrmesh.Vector3f()
    dir.x = 0
    dir.y = 0
    dir.z = 1
    
    undercuts = mrmesh.FaceBitSet()
    mrmesh.findUndercuts(torus, dir, undercuts)
    assert undercuts.count() > 0

    mrmesh.fixUndercuts(torus, dir, 0.2, 0.0)

    assert torus.points.vec.size() > 2900
