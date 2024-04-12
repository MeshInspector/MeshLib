import pytest
from helper import *


def test_self_intersections():
    torusIntersected = mrmesh.makeTorusWithSelfIntersections(2, 1, 10, 10, None)
    selfies = mrmesh.localFindSelfIntersections(torusIntersected)
    assert selfies.count() > 0

    settings = mrmesh.FixSelfIntersectionSettings()
    settings.method = mrmesh.FixSelfIntersectionMethod.CutAndFill
    mrmesh.localFixSelfIntersections(torusIntersected,settings) # just check thit it runs
