import pytest
from helper import *


def test_signed_distance():
    # make torus inside torus
    torus = mrmesh.makeTorus(2, 1, 10, 10, None)
    torus2 = mrmesh.makeTorus(2, 1.2, 10, 10, None)

    xf = mrmesh.AffineXf3f()

    res = mrmesh.findSignedDistance(torus, torus2, xf)
    resRevert = mrmesh.findSignedDistance(torus2, torus, xf)

    # probably, we need negative comparison
    assert res.signedDist == resRevert.signedDist
