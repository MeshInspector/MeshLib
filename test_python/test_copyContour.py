import pytest
from helper import *


def test_copy_contour():
    c3f      = mrmeshpy.Contour3f([mrmeshpy.Vector3f(1.1, 2.2, 3.3), mrmeshpy.Vector3f(4.4, 5.5, 6.6), mrmeshpy.Vector3f(7.7, 8.8, 9.9)])
    c3f_flat = mrmeshpy.Contour3f([mrmeshpy.Vector3f(1.1, 2.2,   0), mrmeshpy.Vector3f(4.4, 5.5,   0), mrmeshpy.Vector3f(7.7, 8.8,   0)])
    c2f      = mrmeshpy.Contour2f([mrmeshpy.Vector2f(1.1, 2.2     ), mrmeshpy.Vector2f(4.4, 5.5     ), mrmeshpy.Vector2f(7.7, 8.8     )])
    c3d      = mrmeshpy.Contour3d([mrmeshpy.Vector3d(1.1, 2.2, 3.3), mrmeshpy.Vector3d(4.4, 5.5, 6.6), mrmeshpy.Vector3d(7.7, 8.8, 9.9)])

    assert mrmeshpy.copyContour_std_vector_Vector3f_std_vector_Vector2f(c2f) == c3f_flat
    assert mrmeshpy.copyContour_std_vector_Vector2f_std_vector_Vector3f(c3f) == c2f
    assert mrmeshpy.copyContour_std_vector_Vector3f_std_vector_Vector3d(c3d) == c3f

    # Now the same thing with a vector of contours.
    assert mrmeshpy.copyContours_std_vector_std_vector_Vector3f_std_vector_std_vector_Vector2f(mrmeshpy.Contours2f([c2f,c2f])) == mrmeshpy.Contours3f([c3f_flat,c3f_flat])
    assert mrmeshpy.copyContours_std_vector_std_vector_Vector2f_std_vector_std_vector_Vector3f(mrmeshpy.Contours3f([c3f,c3f])) == mrmeshpy.Contours2f([c2f,c2f])
    assert mrmeshpy.copyContours_std_vector_std_vector_Vector3f_std_vector_std_vector_Vector3d(mrmeshpy.Contours3d([c3d,c3d])) == mrmeshpy.Contours3f([c3f,c3f])
