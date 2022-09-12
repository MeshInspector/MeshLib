from helper import *
import pytest

def test_fillHole():
    torus = mrmesh.make_torus(2, 1, 10, 10, None)
    
    faceBitSetToDelete = mrmesh.FaceBitSet()
    faceBitSetToDelete.resize(5, False)
    faceBitSetToDelete.set(mrmesh.FaceId(1), True)

    mrmesh.delete_faces(torus.topology, faceBitSetToDelete)
    
    holes = torus.topology.findHoleRepresentiveEdges()
       
    mrmesh.fill_hole(torus, holes[0], mrmesh.FillHoleParams())
    
    assert(torus.topology.findHoleRepresentiveEdges().size() == 0)