from helper import *
import pytest


def test_signedDistance():
# make torus inside torus
    torus = mrmesh.make_torus(2, 1, 10, 10, None)
    torus2 = mrmesh.make_torus(2, 1.2, 10, 10, None)

    xf = mrmesh.AffineXf3()

    res = mrmesh.find_signed_distance(torus, torus2, xf)
    resRevert = mrmesh.find_signed_distance(torus2, torus, xf)

# probably, we need negative comparison
    assert (res.signedDist == resRevert.signedDist)
