import pytest
from helper import *


def test_plane_sections():
    cube = mrmesh.makeCube()
    plane = mrmesh.Plane3f()
    plane.n = mrmesh.Vector3f(0, 0, 1)
    plane.d = 0
    # virtually extract the section without modifying the mesh
    sections = mrmesh.extractPlaneSections(cube, plane)
    contours = mrmesh.planeSectionsToContours2f(cube, sections, mrmesh.AffineXf3f())

    assert sections.size() == 1
    assert contours.size() == 1
    assert contours[0].size() == 9
    assert contours[0][0] == contours[0][8]

    area = mrmesh.calcOrientedArea(contours[0])
    assert abs(area - 1) < 1e-5

    # actually trim the mesh, introducing new vertices and making invalid some old ones
    if is_bindings_v3:
        params = mrmesh.TrimWithPlaneParams()
        params.plane = plane
        mrmesh.trimWithPlane(cube, params)
    else:
        mrmesh.cutMeshWithPlane(cube, plane)

    # check that all valid vertices are on not-negative side of the plane
    validVerts = cube.topology.getValidVerts()
    for v in range(validVerts.size()):
        if validVerts.test(mrmesh.VertId(v)):
            assert cube.points.vec[v].z >= 0
