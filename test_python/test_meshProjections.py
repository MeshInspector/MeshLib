import pytest
from helper import *


def test_mesh_projections():
    torusRef = mrmesh.makeTorus(2, 1, 10, 10, None)
    torus = mrmesh.makeTorus(2, 1, 10, 10, None)
    torus.transform(mrmesh.AffineXf3f.translation(mrmesh.Vector3f(0.5, 1, 1)))

    unsignedDist = mrmesh.findProjection(mrmesh.Vector3f(0, 0, 0), torusRef)
    assert unsignedDist.distSq < 10.0
    signedDist = mrmesh.findSignedDistance(mrmesh.Vector3f(1.5, 0, 0), torusRef)
    assert signedDist.dist < 0.0

    allProjRes = mrmesh.projectAllMeshVertices(torusRef, torus)
    assert allProjRes.vec.size() == torus.points.vec.size()
    for i in allProjRes.vec:
        assert i < 10
