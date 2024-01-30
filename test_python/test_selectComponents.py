import pytest
from helper import *


def test_select_components():
    torus = mrmesh.makeTorusWithComponents(2, 1, 10, 10, None)

    components = mrmesh.getAllComponentsVerts(torus, None)

    assert len(components) == 5

    assert components[0].count() == components[1].count()
    assert components[1].count() == components[2].count()
    assert components[2].count() == components[3].count()
    assert components[3].count() == components[4].count()
    assert components[4].count() == components[0].count()
