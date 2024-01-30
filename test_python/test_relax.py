import pytest
from helper import *


def test_relax():
    R1 = 2
    R2_1 = 1
    R2_2 = 2.5
    torus = mrmesh.makeTorusWithSpikes(R1, R2_1, R2_2, 10, 12, None)

    params = mrmesh.MeshRelaxParams()
    params.iterations = 5
    res = mrmesh.relax(torus, params)

    assert res


def test_relax_keep_volume():
    R1 = 2
    R2_1 = 1
    R2_2 = 2.5
    keep_volume_torus = mrmesh.makeTorusWithSpikes(R1, R2_1, R2_2, 10, 12, None)

    params = mrmesh.MeshRelaxParams()
    params.iterations = 5
    res = mrmesh.relaxKeepVolume(keep_volume_torus, params)

    assert res

def test_relax_approx():
    R1 = 2
    R2_1 = 1
    R2_2 = 2.5
    torus = mrmesh.makeTorusWithSpikes(R1, R2_1, R2_2, 10, 12, None)

    params = mrmesh.MeshApproxRelaxParams()
    params.iterations = 5
    res = mrmesh.relaxApprox(torus, params)

    assert res

def test_smooth_region_boundary():
    R1 = 2
    R2_1 = 1
    R2_2 = 2.5
    keep_volume_torus = mrmesh.makeTorusWithSpikes(R1, R2_1, R2_2, 10, 12, None)

    smooth_region = mrmesh.FaceBitSet()
    smooth_region.resize( 6, True )

    # This just checks that the function exists and can be called.
    mrmesh.smoothRegionBoundary(keep_volume_torus, smooth_region)
