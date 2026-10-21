import pytest
from helper import *


def test_vector_len():
    points = mrmesh.VertCoords()
    assert len(points) == 0
    assert not points
    points.push_back(mrmesh.Vector3f(1, 2, 3))
    assert len(points) == 1
    assert points


def test_vector_from_list():
    points = mrmesh.VertCoords([mrmesh.Vector3f(0, 0, 0), mrmesh.Vector3f(1, 0, 0), mrmesh.Vector3f(0, 1, 0)])
    assert len(points) == 3
    assert points[mrmesh.VertId(1)].x == 1

    scalars = mrmesh.VertScalars([1.0, 3.0, 2.0])
    assert len(scalars) == 3
    assert max(scalars) == 3.0

    with pytest.raises(Exception):
        mrmesh.VertCoords([1, 2])


def test_vector_list_argument():
    # a list is implicitly converted to Triangulation
    topology = mrmesh.topologyFromTriangles([mrmesh.ThreeVertIds([mrmesh.VertId(0), mrmesh.VertId(1), mrmesh.VertId(2)])])
    assert topology.numValidFaces() == 1
    assert topology.numValidVerts() == 3
