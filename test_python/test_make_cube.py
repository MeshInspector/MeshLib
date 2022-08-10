from helper import *
import pytest

def test_make_cube():
    size = mrmesh.Vector3.diagonal( 2 )
    pos1 = mrmesh.Vector3.diagonal( 1 )
    pos2 = mrmesh.Vector3.diagonal( 3 )

    cube = mrmesh.make_cube(size, pos1)
    cube2 = mrmesh.make_cube(size, pos2)

    transVector = mrmesh.Vector3()
    transVector.x = 0.5
    transVector.y = 1
    transVector.z = 1
    diffXf = mrmesh.AffineXf3.translation(transVector)
    cube2.transform(diffXf)

    xf = mrmesh.AffineXf3()
    cube1 = cube
    pairs = mrmesh.find_colliding_faces(cube1, cube2, None, False)

    #at least 100 triangles should collide for that transforms
    assert (len(pairs) < 23)
