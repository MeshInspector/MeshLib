import pytest
from helper import *


def is_equal_vector3(a, b):
    diff = a - b
    return diff.length() < 1.0e-6


def test_boolean_exposing():
    size = mrmesh.Vector3f.diagonal(2)
    pos1 = mrmesh.Vector3f.diagonal(0)
    pos2 = mrmesh.Vector3f.diagonal(-1)
    pos3 = mrmesh.Vector3f.diagonal(1)

    meshA = mrmesh.makeCube(size, pos1)
    meshB = mrmesh.makeCube(size, pos2)

    bOperation = mrmesh.BooleanOperation.Intersection
    bResMapper = mrmesh.BooleanResultMapper()
    bResult = mrmesh.boolean(meshA, meshB, bOperation, None, bResMapper)
    assert bResult.valid(), bResult.errorString

    bResMesh = bResult.mesh

    assert is_equal_vector3(
        bResMesh.computeBoundingBox(
            bResMesh.topology.getValidFaces(), mrmesh.AffineXf3f()
        ).min,
        pos1,
    )
    assert is_equal_vector3(
        bResMesh.computeBoundingBox(
            bResMesh.topology.getValidFaces(), mrmesh.AffineXf3f()
        ).max,
        pos3,
    )

    assert bResMesh.topology.getValidVerts().size() == 14
    assert bResMesh.topology.getValidVerts().count() == 14
    assert bResMesh.topology.findHoleRepresentiveEdges().size() == 0

    brmmAA = bResMapper.map(meshA.topology.getValidVerts(), mrmesh.BooleanResMapObj.A)
    brmmBB = bResMapper.map(meshB.topology.getValidVerts(), mrmesh.BooleanResMapObj.B)

    filteredOldFacesA = bResMapper.filteredOldFaceBitSet( meshA.topology.getValidFaces(), mrmesh.BooleanResMapObj.A )
    mapsA = bResMapper.getMaps(mrmesh.BooleanResMapObj.A)
    assert mapsA.cut2newFaces.vec.size() == 42
    assert brmmAA.count() == 1
    assert brmmBB.count() == 1
    assert filteredOldFacesA.count() == 6


def test_unite_many_meshes():
    size = mrmesh.Vector3f.diagonal(2)
    poses = [
        mrmesh.Vector3f.diagonal(-1),
        mrmesh.Vector3f.diagonal(0),
        mrmesh.Vector3f.diagonal(1),
        mrmesh.Vector3f.diagonal(2),
    ]
    meshes = []
    vecMeshes = mrmesh.vectorConstMeshPtr()
    vecMeshes.resize(len(poses))
    for i in range(len(poses)):
        meshes.append(mrmesh.makeCube(size, poses[i]))
        vecMeshes[i] = meshes[i]
    params = mrmesh.UniteManyMeshesParams()
    params.nestedComponentsMode = mrmesh.NestedComponenetsMode.Remove
    resMesh = mrmesh.uniteManyMeshes(vecMeshes)
    assert resMesh.topology.numValidFaces() > 0
    assert resMesh.topology.findHoleRepresentiveEdges().size() == 0

def test_intersection_contours():
    size = mrmesh.Vector3f.diagonal(2)
    pos1 = mrmesh.Vector3f.diagonal(0)
    pos2 = mrmesh.Vector3f.diagonal(-1)
    pos3 = mrmesh.Vector3f.diagonal(1)

    meshA = mrmesh.makeCube(size, pos1)
    meshB = mrmesh.makeCube(size, pos2)

    conv = mrmesh.getVectorConverters(meshA,meshB)
    if is_bindings_v3:
        intersections = mrmesh.findCollidingEdgeTrisPrecise(meshA,meshB,conv.toInt)
    else:
        intersections = mrmesh.findCollidingEdgeTrisPrecise(meshA,meshB,conv)
    orderedIntersections = mrmesh.orderIntersectionContours(meshA.topology,meshA.topology,intersections)

    aConts = mrmesh.OneMeshContours()
    bConts = mrmesh.OneMeshContours()
    mrmesh.getOneMeshIntersectionContours(meshA,meshB,orderedIntersections,aConts,bConts,conv)
    assert aConts[0].intersections.size() > 0
    assert aConts[0].closed
    assert aConts[0].intersections.size() == bConts[0].intersections.size()
    assert bConts[0].closed