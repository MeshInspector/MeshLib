from helper import *
import pytest



def test_delete_faces():
    torus = mrmesh.make_torus(2, 1, 10, 10, None)
    
    
    faceBitSetToDelete = mrmesh.FaceBitSet()
    faceBitSetToDelete.resize(5, False)
    faceBitSetToDelete.set(mrmesh.FaceId(1), True)
    oldFaceBS = torus.topology.getValidFaces()
    mrmesh.delete_faces(torus.topology, faceBitSetToDelete)
    
    deletedBitSet = oldFaceBS - torus.topology.getValidFaces()
  
    assert(deletedBitSet.count() == 1)
    assert(deletedBitSet.test(mrmesh.FaceId(1)))
    
    
    
    
    #delete_faces(...) method of builtins.PyCapsule instance
        #delete_faces(arg0: mrmeshpy.MeshTopology, arg1: mrmeshpy.FaceBitSet) -> None