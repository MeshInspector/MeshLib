from helper import *
import pytest


def test_bitSetTest():
    torus = mrmesh.makeTorusWithComponents(2, 1, 10, 10, None)

    components = mrmesh.getAllComponentsVerts(torus, None)

    assert(len(components) == 5)
    assert(components[0].count() != 0)
    comp0Flip = mrmesh.VertBitSet()
    comp0Flip |= components[0]
    comp0Flip.flip()

    assert(comp0Flip.size() == components[0].size())
    assert(comp0Flip.count() == (components[0].size() - components[0].count()))

    xorRes = comp0Flip ^ components[0]

    assert(xorRes.size() == components[0].size())
    assert(xorRes.count() == components[0].size())