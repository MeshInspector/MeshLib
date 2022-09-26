from helper import *
import pytest


def isEqualVector3(a, b):
    diff = a - b
    return diff.length() < 1.e-6

def test_booleanExposing():
    size = mrmesh.Vector3f.diagonal( 2 )
    pos1 = mrmesh.Vector3f.diagonal( 0 )
    pos2 = mrmesh.Vector3f.diagonal( -1 )
    pos3 = mrmesh.Vector3f.diagonal( 1 )

    meshA = mrmesh.makeCube(size, pos1)
    meshB = mrmesh.makeCube(size, pos2)

    bOperation = mrmesh.BooleanOperation.Intersection
    bResMapper = mrmesh.BooleanResultMapper()
    bResult = mrmesh.boolean( meshA, meshB, bOperation, None, bResMapper )

    bResMesh = bResult.mesh

    assert( isEqualVector3( bResMesh.computeBoundingBox(bResMesh.topology.getValidFaces(), mrmesh.AffineXf3f() ).min , pos1 ) )
    assert( isEqualVector3( bResMesh.computeBoundingBox(bResMesh.topology.getValidFaces(), mrmesh.AffineXf3f() ).max , pos3 ) )

    assert( bResMesh.topology.getValidVerts().size() == 14 )
    assert( bResMesh.topology.getValidVerts().count() == 14 )
    assert( bResMesh.topology.findHoleRepresentiveEdges().size() == 0 )


    brmmAA = bResMapper.map( meshA.topology.getValidVerts(), mrmesh.BooleanResMapObj.A )
    brmmBB = bResMapper.map( meshB.topology.getValidVerts(), mrmesh.BooleanResMapObj.B )

    assert( brmmAA.count() == 1)
    assert( brmmBB.count() == 1)