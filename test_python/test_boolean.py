from helper import *
import pytest


def test_boolean():
    torusIntersected = mrmesh.makeTorusWithSelfIntersections(2, 1, 10, 10, None)
    mrmesh.fixSelfIntersections(torusIntersected, 0.1)

    torus = mrmesh.makeTorus(2, 1, 10, 10, None)

    transVector = mrmesh.Vector3f()
    transVector.x=0.5
    transVector.y=1
    transVector.z=1

    diffXf = mrmesh.AffineXf3f.translation(transVector)

    torus2 = mrmesh.makeTorus(2, 1, 10, 10, None)
    torus2.transform(diffXf)

    p = torus.points.vec.size()

    torusS = mrmesh.voxelBooleanSubtract(torus, torus2, 0.05)
    p_sub = torusS.points.vec.size()

    torusU = mrmesh.voxelBooleanUnite(torus, torus2, 0.05)
    p_union = torusU.points.vec.size()

    torusI = mrmesh.voxelBooleanIntersect(torus, torus2, 0.05)
    p_intersect = torusI.points.vec.size()

    assert( p == 100)
    assert( p_sub == 43132)
    assert( p_union == 63114)
    assert( p_intersect == 23006)
