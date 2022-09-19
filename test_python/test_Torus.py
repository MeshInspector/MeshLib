from helper import *
import pytest

def test_torus():
    torus = mrmesh.makeOuterHalfTorus(2, 1, 10, 10, None)
    #mrmesh.save_mesh(torus, "/home/tim/models/testTorus_half.stl")

    torus = mrmesh.makeTorusWithUndercut(2, 1, 1.5, 10, 10, None)
    #mrmesh.save_mesh(torus, "/home/tim/models/testTorus_undercut.stl")

    torus = mrmesh.makeTorusWithSpikes(2, 1, 2.5, 10, 10, None)
    #mrmesh.save_mesh(torus, "/home/tim/models/testTorus_spikes.stl")

    torus = mrmesh.makeTorusWithComponents(2, 1, 10, 10, None)
    #mrmesh.save_mesh(torus, "/home/tim/models/testTorus_components.stl")

    torus = mrmesh.makeTorusWithSelfIntersections(2, 1, 10, 10, None)
    #mrmesh.save_mesh(torus, "/home/tim/models/testTorus_selfintersect.stl")
