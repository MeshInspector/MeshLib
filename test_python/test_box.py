import pytest
from helper import *


def test_box():
    b = mrmesh.Box2f()
    assert b.valid() == False

    val0 = 0.0
    v0 = mrmesh.Vector2f.diagonal(val0)
    b.min = v0
    b.max = v0
    assert b.valid() == True
    assert b.min == v0
    assert b.max == v0
    assert b.center() == v0
    assert b.size() == v0
    assert b.diagonal() == 0.0
    assert b.volume() == 0.0

    val1 = 1.0
    v1 = mrmesh.Vector2f.diagonal(val1)
    b.include(v1)
    assert b.valid() == True
    assert b.min == v0
    assert b.max == v1
    assert b.center() == ((v0 + v1) / 2.0)
    assert b.size() == v1
    assert abs(b.diagonal() - 2.0**0.5) < 1.0e-6
    assert b.volume() == 1.0
    assert b.contains(v0) == True
    assert b.contains(v1) == True
    assert b.contains((v0 + v1) / 2.0) == True
    assert b.contains(-v1) == False
    assert b.contains(v0 - v1) == False
    assert b.contains(v1 + v1) == False

    val2 = 2.0
    val3 = 3.0
    v2 = mrmesh.Vector2f.diagonal(val2)
    v3 = mrmesh.Vector2f.diagonal(val3)
    b2 = mrmesh.Box2f()
    b2.min = v2
    b2.max = v3
    assert b2.valid() == True
    assert b2.min == v2
    assert b2.max == v3
    assert b2.center() == ((v2 + v3) / 2.0)
    assert b2.size() == v1
    assert abs(b2.diagonal() - 2.0**0.5) < 1.0e-6
    assert b2.volume() == 1.0
    assert b2.contains(v2) == True
    assert b2.contains(v3) == True
    assert b2.contains((v2 + v3) / 2.0) == True
    assert b2.contains(v2 - v1) == False
    assert b2.contains(v3 + v1) == False
    assert b.intersects(b2) == False

    b.include(b2)
    assert b.valid() == True
    assert b.min == v0
    assert b.max == v3
    assert b.center() == ((v0 + v3) / 2.0)
    assert b.size() == v3
    assert abs(b.diagonal() - (val3 - val0) * 2.0**0.5) < 1.0e-6
    assert b.volume() == (val3 - val0) ** 2

    b3 = mrmesh.Box2f()
    b3.min = v0
    b3.max = v2
    b4 = mrmesh.Box2f()
    b4.min = v1
    b4.max = v3
    assert b3.intersects(b4) == True
    b5 = b3.intersects(b4)

    b5 = mrmesh.Box2f()
    b5.min = v1
    b5.max = v2

def test_densebox():
    mesh = mrmesh.makeSphere(mrmesh.SphereParams())
    mesh.transform( mrmesh.AffineXf3f.linear(mrmesh.Matrix3f.rotation(mrmesh.Vector3f(1,0,0),mrmesh.Vector3f(1,1,1)))* mrmesh.AffineXf3f.linear(mrmesh.Matrix3f.scale(1,2,3)) )
    denseBox = mrmesh.DenseBox(mesh)
    denseCube = mrmesh.makeCube(denseBox.box().size(),denseBox.box().min)
    denseCube.transform(denseBox.basisXf())

    assert denseCube.volume() < mesh.computeBoundingBox().volume()