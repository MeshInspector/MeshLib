from meshlib import mrmeshpy as mm
import pytest

@pytest.mark.bindingsV3
@pytest.mark.smoke
def test_matrix3f():
    v1 = mm.Vector3f(1, 2, 3)
    v2 = mm.Vector3f(3, 4, 5)
    v3 = mm.Vector3f(4, 5, 6)

    a = mm.Matrix3f().fromColumns(v1, v2, v3)
    b = mm.Matrix3f().fromRows(v1, v2, v3)
    c = a * b
    print(c.col(0) == mm.Vector3f(26, 34, 42))
    print(c.col(1) == mm.Vector3f(34, 45, 56))
    print(c.col(2) == mm.Vector3f(42, 56, 70))

@pytest.mark.bindingsV3
@pytest.mark.smoke
def test_matrix3i():
    v1 = mm.Vector3i(1, 2, 3)
    v2 = mm.Vector3i(3, 4, 5)
    v3 = mm.Vector3i(4, 5, 6)

    a = mm.Matrix3i().fromColumns(v1, v2, v3)
    b = mm.Matrix3i().fromRows(v1, v2, v3)
    c = a * b
    print(c.col(0) == mm.Vector3i(26, 34, 42))
    print(c.col(1) == mm.Vector3i(34, 45, 56))
    print(c.col(2) == mm.Vector3i(42, 56, 70))
