import pytest
from helper import *


def test_bit_settest():
    torus = mrmesh.makeTorusWithComponents(2, 1, 10, 10, None)

    components = mrmesh.getAllComponentsVerts(torus, None)

    assert len(components) == 5
    assert components[0].count() != 0

    counter = 0
    for v in components[0]:
        assert components[0].test( v )
        counter = counter + 1
    assert counter == components[0].count()

    comp0Flip = mrmesh.VertBitSet()
    comp0Flip |= components[0]
    comp0Flip.flip()

    assert comp0Flip.size() == components[0].size()
    assert comp0Flip.count() == (components[0].size() - components[0].count())

    xorRes = comp0Flip ^ components[0]

    assert xorRes.size() == components[0].size()
    assert xorRes.count() == components[0].size()
