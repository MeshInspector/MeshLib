from helper import *
import pytest


def test_icp():
    torusRef = mrmesh.make_torus(2,1,32,32,None)
    torusMove = mrmesh.make_torus(2,1,32,32,None)

    axis = mrmesh.Vector3()
    axis.x=1

    trans = mrmesh.Vector3()
    trans.y=0.2
    trans.z=0.105

    xf = mrmesh.AffineXf3.translation(trans) * mrmesh.AffineXf3.linear( mrmesh.Matrix3.rotation( axis, 0.2 ) )

    icp = mrmesh.MeshICP(torusMove,torusRef,xf,mrmesh.AffineXf3(),torusMove.topology.getValidVerts())
    newXf = icp.calculateTransformation()

    diffXf = mrmesh.AffineXf3()
    diffXf.A -= newXf.A
    diffXf.b -= newXf.b

    assert(abs(diffXf.A.x.x)<1e-6 )
    assert(abs(diffXf.A.x.y)<1e-6 )
    assert(abs(diffXf.A.x.z)<1e-6 )

    assert(abs(diffXf.A.y.x)<1e-6 )
    assert(abs(diffXf.A.y.y)<1e-6 )
    assert(abs(diffXf.A.y.z)<1e-6 )

    assert(abs(diffXf.A.z.x)<1e-6 )
    assert(abs(diffXf.A.z.y)<1e-6 )
    assert(abs(diffXf.A.z.z)<1e-6 )

    assert(abs(diffXf.b.x)<1e-6 )
    assert(abs(diffXf.b.y)<1e-6 )
    assert(abs(diffXf.b.z)<1e-6 )