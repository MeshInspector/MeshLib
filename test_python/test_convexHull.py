from helper import *
import pytest


def test_convexHull():
    torus = mrmesh.makeTorus(2, 1, 10, 10, None)

    # Create the convex hull
    hull = mrmesh.convexHull(torus)

    # Check that all vertives of the torus are inside the hull
    outside = mrmesh.boolean(torus, hull, mrmesh.BooleanOperation.OutsideB).mesh
    
    # There should be no vertices outside the hull
    assert outside.points.size() == 0