import pytest
from helper import *


def test_movement_body():
    inputMesh = mrmesh.makeCube()
    box = inputMesh.computeBoundingBox()
    step = (box.max.z - box.min.z)/11
    start = box.min.z + step/2

    bodyContours = mrmesh.Contours3f()
    bodyContours.resize(1)
    bodyContours[0].resize(5)
    bodyContours[0][0] = mrmesh.Vector3f(-0.01,-0.01,0)
    bodyContours[0][1] = mrmesh.Vector3f(0.01,-0.01,0)
    bodyContours[0][2] = mrmesh.Vector3f(0.01,0.01,0)
    bodyContours[0][3] = mrmesh.Vector3f(-0.01,0.01,0)
    bodyContours[0][4] = bodyContours[0][0]

    sumMesh = mrmesh.Mesh()

    while start < box.max.z:
        plane = mrmesh.Plane3f()
        plane.n = mrmesh.Vector3f(0,0,1)
        plane.d = start
        plSections = mrmesh.extractPlaneSections(inputMesh, plane)
        contours = mrmesh.Contours3f()
        contours.resize( plSections.size() )
        for i in range(len(plSections)):
            contours[i].resize(plSections[i].size())
            for j in range(plSections[i].size()):
                contours[i][j] = inputMesh.edgePoint( plSections[i][j] )
        moveMesh = mrmesh.makeMovementBuildBody(bodyContours,contours)
        sumMesh.addPartByMask(moveMesh,moveMesh.topology.getValidFaces())
        start = start + step;
        pass

    assert mrmesh.getAllComponents(sumMesh).size() == 11
    assert sumMesh.topology.findHoleRepresentiveEdges().size() == 0
