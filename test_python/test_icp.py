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

    # eps increase from 1e-6 to 3e-5 for GCC11 Debug Arm build
    eps = 3e-5

    assert abs(diffXf.A.x.x) < eps
    assert abs(diffXf.A.x.y) < eps
    assert abs(diffXf.A.x.z) < eps

    assert abs(diffXf.A.y.x) < eps
    assert abs(diffXf.A.y.y) < eps
    assert abs(diffXf.A.y.z) < eps

    assert abs(diffXf.A.z.x) < eps
    assert abs(diffXf.A.z.y) < eps
    assert abs(diffXf.A.z.z) < eps

    assert abs(diffXf.b.x) < eps
    assert abs(diffXf.b.y) < eps
    assert abs(diffXf.b.z) < eps
