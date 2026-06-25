import pytest
from helper import *


def test_icp():
    torusRef = mrmesh.makeTorus(2.5, 0.7, 48, 48, None)
    torusMove = mrmesh.makeTorus(2.5, 0.7, 48, 48, None)

    axis = mrmesh.Vector3f()
    axis.x = 1

    trans = mrmesh.Vector3f()
    trans.y = 0.2
    trans.z = 0.105

    xf = mrmesh.AffineXf3f.translation(trans) * mrmesh.AffineXf3f.linear(
        mrmesh.Matrix3f.rotation(axis, 0.2)
    )

    icp = mrmesh.ICP(
        torusMove, torusRef, xf, mrmesh.AffineXf3f(), torusMove.topology.getValidVerts(), torusRef.topology.getValidVerts()
    )

    props = mrmesh.ICPProperties()
    props.iterLimit = 20
    icp.setParams(props)
    newXf = icp.calculateTransformation()
    print(icp.getStatusInfo())

    diffXf = mrmesh.AffineXf3f()
    diffXf.A -= newXf.A
    diffXf.b -= newXf.b

    assert abs(diffXf.A.x.x) < 1e-6
    assert abs(diffXf.A.x.y) < 1e-6
    assert abs(diffXf.A.x.z) < 1e-6

    assert abs(diffXf.A.y.x) < 1e-6
    assert abs(diffXf.A.y.y) < 1e-6
    assert abs(diffXf.A.y.z) < 1e-6

    assert abs(diffXf.A.z.x) < 1e-6
    assert abs(diffXf.A.z.y) < 1e-6
    assert abs(diffXf.A.z.z) < 1e-6

    assert abs(diffXf.b.x) < 1e-6
    assert abs(diffXf.b.y) < 1e-6
    assert abs(diffXf.b.z) < 1e-6
