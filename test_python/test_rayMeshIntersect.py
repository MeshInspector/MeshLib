from helper import *
import pytest


def test_ray_mesh_intersect():
    torus = mrmesh.makeTorus(2, 1, 10, 10, None)
    interRes1 = mrmesh.rayMeshIntersect(torus,mrmesh.Line3f(mrmesh.Vector3f(),mrmesh.Vector3f(0,0,1)));
    interRes2 = mrmesh.rayMeshIntersect(torus,mrmesh.Line3f(mrmesh.Vector3f(),mrmesh.Vector3f(0.5,0,0)));
    assert (not interRes1)
    assert (interRes2 and interRes2.proj.point == mrmesh.Vector3f(1,0,0))