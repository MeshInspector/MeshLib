import shutil

import pytest
from helper import *


def test_segmentation():
    mesh = mrmesh.makeTorus()
    metric = mrmesh.edgeLengthMetric(mesh)
    source = mrmesh.FaceBitSet()
    sink = mrmesh.FaceBitSet()
    source.resize(mesh.topology.getValidFaces().size(), False)
    sink.resize(mesh.topology.getValidFaces().size(), False)

    source.set(mrmesh.FaceId(0), True)
    sink.set(mrmesh.FaceId(5), True)

    res = mrmesh.segmentByGraphCut(mesh.topology, source, sink, metric)

    assert res.count() != 0
    assert res.test(mrmesh.FaceId(0))
    assert not res.test(mrmesh.FaceId(5))
