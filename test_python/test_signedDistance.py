from helper import *
import pytest


def test_signedDistance():
# make torus inside torus
    torus = mrmesh.makeTorus(2, 1, 10, 10, None)
    torus2 = mrmesh.makeTorus(2, 1.2, 10, 10, None)

    xf = mrmesh.AffineXf3f()

    res = mrmesh.findSignedDistance(mrmesh.MeshPart(torus), mrmesh.MeshPart(torus2), xf)
    resRevert = mrmesh.findSignedDistance(mrmesh.MeshPart(torus2),mrmesh.MeshPart( torus), xf)

# probably, we need negative comparison
    assert (res.signedDist == resRevert.signedDist)
