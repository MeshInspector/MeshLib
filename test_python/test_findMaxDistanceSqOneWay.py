from helper import *
import pytest

def test_findMaxDistanceSqOneWay():
    torus = mrmesh.make_torus(2, 1, 10, 10, None)
    torus2 = mrmesh.make_torus(2, 1, 10, 10, None)

    transVector = mrmesh.Vector3f()
    transVector.x = 40
    transVector.y = 40
    transVector.z = 40
    diffXf = mrmesh.AffineXf3f.translation(transVector)
    torus2.transform(diffXf)

    distSq = mrmesh.findMaxMeshDistanceSqOneWay(mrmesh.MeshPart(torus), mrmesh.MeshPart(torus2), diffXf.inverse(), 1e35)
    
    assert (distSq < 69.28205**2)