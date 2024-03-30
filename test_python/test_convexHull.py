import pytest
from helper import *


def test_convexHull():
    torus = mrmesh.makeTorus(2, 1, 10, 10, None)

    # Create the convex hull
    hull = mrmesh.makeConvexHull(torus)

    # test hull volume
    assert abs(hull.volume() - 43.184) < 0.01
