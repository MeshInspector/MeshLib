from helper import *
import pytest

def test_Vector3():
    double_vec = mrmesh.Vector3d(1, 2, 3)
    int_vec = mrmesh.Vector3i(1, 2, 3)
    float_vec = mrmesh.Vector3f(1, 2, 3)

    doubles_list = list(double_vec)
    ints_list = list(int_vec)
    floats_list = list(float_vec)

    assert len(doubles_list) == 3
    assert len(ints_list) == 3
    assert len(floats_list) == 3

    assert type(doubles_list[0]) == float # Double is converted to float in python
    assert type(ints_list[0]) == int
    assert type(floats_list[0]) == float

def test_Vector2():
    double_vec = mrmesh.Vector2d(1, 2)
    int_vec = mrmesh.Vector2i(1, 2)
    float_vec = mrmesh.Vector2f(1, 2)

    doubles_list = list(double_vec)
    ints_list = list(int_vec)
    floats_list = list(float_vec)

    assert len(doubles_list) == 2
    assert len(ints_list) == 2
    assert len(floats_list) == 2

    assert type(doubles_list[0]) == float # Double is converted to float in python
    assert type(ints_list[0]) == int
    assert type(floats_list[0]) == float