from meshlib import mrmeshpy
import pytest

@pytest.mark.smoke
@pytest.mark.bindingsV3
def test_string_reporting():
    vector = mrmeshpy.Vector3f(1, 2, 3)
    assert str(vector) == "Vector3f(1 2 3)"
