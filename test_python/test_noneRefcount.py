import sys

import pytest
from helper import *


def _construct():
    mrmesh.Vector3f(1.0, 2.0, 3.0)


def _set_attribute():
    v = mrmesh.Vector3f()
    for _ in range(10):
        v.x = 1.0


def _set_item():
    c = mrmesh.Contour3f()
    c.resize(1)
    v = mrmesh.Vector3f()
    for _ in range(10):
        c[0] = v


# Bindings built with the Python 3.12 headers dropped a reference to `None` in each of these calls on Python 3.8-3.11
#   until it was deallocated (#7510). Since 3.12 `None` is immortal, and the check passes trivially.
@pytest.mark.parametrize("op", [_construct, _set_attribute, _set_item])
def test_none_refcount(op):
    n = 1000
    op()
    before = sys.getrefcount(None)
    for _ in range(n):
        op()
    lost = before - sys.getrefcount(None)
    assert lost < n // 2, f"{op.__name__} lost {lost} references to None in {n} calls"
