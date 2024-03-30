import pytest
from helper import *


def test_subdivider():
    torus = mrmesh.makeTorus(2, 1, 10, 10, None)
    countInit = torus.topology.getValidFaces().count()

    settings = mrmesh.SubdivideSettings()
    settings.maxEdgeLen = 0.3
    settings.maxEdgeSplits = 5
    settings.maxDeviationAfterFlip = 0.2
    settings.region = None
    mrmesh.subdivideMesh(torus, settings)

    assert torus.topology.getValidFaces().count() > countInit
