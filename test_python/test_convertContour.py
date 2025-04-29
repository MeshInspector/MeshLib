import pytest
from helper import *


def test_copy_contour():
    c3f      = mrmesh.Contour3f([mrmesh.Vector3f(1.1, 2.2, 3.3), mrmesh.Vector3f(4.4, 5.5, 6.6), mrmesh.Vector3f(7.7, 8.8, 9.9)])
    c3f_flat = mrmesh.Contour3f([mrmesh.Vector3f(1.1, 2.2,   0), mrmesh.Vector3f(4.4, 5.5,   0), mrmesh.Vector3f(7.7, 8.8,   0)])
    c2f      = mrmesh.Contour2f([mrmesh.Vector2f(1.1, 2.2     ), mrmesh.Vector2f(4.4, 5.5     ), mrmesh.Vector2f(7.7, 8.8     )])
    c3d      = mrmesh.Contour3d([mrmesh.Vector3d(1.1, 2.2, 3.3), mrmesh.Vector3d(4.4, 5.5, 6.6), mrmesh.Vector3d(7.7, 8.8, 9.9)])

    assert mrmesh.convertContourTo3f(c2f) == c3f_flat
    assert mrmesh.convertContourTo2f(c3f) == c2f
    assert mrmesh.convertContourTo3f(c3d) == c3f

    # Now the same thing with a vector of contours.
    assert mrmesh.convertContoursTo3f(mrmesh.Contours2f([c2f,c2f])) == mrmesh.Contours3f([c3f_flat,c3f_flat])
    assert mrmesh.convertContoursTo2f(mrmesh.Contours3f([c3f,c3f])) == mrmesh.Contours2f([c2f,c2f])
    assert mrmesh.convertContoursTo3f(mrmesh.Contours3d([c3d,c3d])) == mrmesh.Contours3f([c3f,c3f])
