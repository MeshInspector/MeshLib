import pytest
from helper import *


def test_plane_sections():
    cube = mrmesh.makeCube()
    plane = mrmesh.Plane3f()
    plane.n = mrmesh.Vector3f(0, 0, 1)
    plane.d = 0
    sections = mrmesh.extractPlaneSections(cube, plane)
    contours = mrmesh.planeSectionsToContours2f(cube, sections, mrmesh.AffineXf3f())

    assert sections.size() == 1
    assert len(contours) == 1
    assert len(contours[0]) == 9
    assert contours[0][0] == contours[0][8]

    area = mrmesh.calcOrientedArea(contours[0])
    assert abs(area - 1) < 1e-5
