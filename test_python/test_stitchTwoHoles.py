from helper import *
import pytest

def test_stitch_TwoHoles():
    torus = mrmesh.makeOuterHalfTorus(2, 1, 10, 10, None)
    torus2 = mrmesh.makeOuterHalfTorus(2, 1, 10, 10, None)
    torusAutostitchTwo = torus

    holes = torus.topology.findHoleRepresentiveEdges()
    assert(len(holes)==2)

    params = mrmesh.StitchHolesParams()
    mrmesh.set_stitch_holes_metric_edge_length(params, torus)
    mrmesh.stitch_holes(torus, holes[0], holes[1], params)

    holes = torus.topology.findHoleRepresentiveEdges()
    assert(len(holes)==0)

    mrmesh.stitch_two_holes(torus2, params)
    holes = torus2.topology.findHoleRepresentiveEdges()
    assert(len(holes)==0)