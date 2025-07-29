import pytest
from helper import *


def test_fix_undercuts():
    torus = mrmesh.makeTorusWithUndercut(2, 1, 1.5, 10, 10, None)

    params = mrmesh.FixUndercuts.FixParams()
    params.findParameters.upDirection = mrmesh.Vector3f(0, 0, 1)
    params.voxelSize = 0.2

    undercuts = mrmesh.FaceBitSet()
    mrmesh.FixUndercuts.find(torus, params.findParameters, undercuts)
    assert undercuts.count() > 0

    mrmesh.FixUndercuts.fix(torus, params)

    assert torus.points.vec.size() > 2900
