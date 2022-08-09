from helper import *
import pytest

def test_positionVersSmooth():
    R1 = 2
    R2_1 = 1
    R2_2 = 2.5
    torus = mrmesh.make_spikes_test_torus(R1, R2_1, R2_2, 10, 12, None)
    #torus = mrmesh.make_torus(R1, R2_2, 10, 10, None)

    #params = mrmesh.LaplacianEdgeWeightsParam.Unit
    params = mrmesh.LaplacianEdgeWeightsParam.Cotan
    verts = torus.topology.getValidVerts()
    mrmesh.position_verts_smoothly(torus, verts, params)

    # now all points are in that range from the center
    for i in torus.points.vec:
        assert (i.x*i.x + i.y*i.y + i.z*i.z == 0. )
