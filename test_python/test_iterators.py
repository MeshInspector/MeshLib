from typing import List, Union

import pytest
from helper import *


def elementwise_comparison_3(
    a: List, b: Union[mrmesh.Vector3d, mrmesh.Vector3i, mrmesh.Vector3f]
):
    assert a[0] == b.x
    assert a[1] == b.y
    assert a[2] == b.z


def elementwise_comparison_2(
    a: List, b: Union[mrmesh.Vector2d, mrmesh.Vector2i, mrmesh.Vector2f]
):
    assert a[0] == b.x
    assert a[1] == b.y


def iteration_check(
    a: Union[
        mrmesh.Vector3d,
        mrmesh.Vector3i,
        mrmesh.Vector3f,
        mrmesh.Vector2d,
        mrmesh.Vector2i,
        mrmesh.Vector2f,
    ]
):
    counter = 1
    for el in a:
        assert el == counter
        counter = counter + 1


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

    assert type(doubles_list[0]) == float  # Double is converted to float in python
    assert type(ints_list[0]) == int
    assert type(floats_list[0]) == float

    elementwise_comparison_3(doubles_list, double_vec)
    elementwise_comparison_3(ints_list, int_vec)
    elementwise_comparison_3(floats_list, float_vec)

    iteration_check(double_vec)
    iteration_check(int_vec)
    iteration_check(float_vec)


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

    assert type(doubles_list[0]) == float  # Double is converted to float in python
    assert type(ints_list[0]) == int
    assert type(floats_list[0]) == float

    elementwise_comparison_2(doubles_list, double_vec)
    elementwise_comparison_2(ints_list, int_vec)
    elementwise_comparison_2(floats_list, float_vec)

    iteration_check(double_vec)
    iteration_check(int_vec)
    iteration_check(float_vec)

def test_ValidFacesIteration():
    mesh = mrmesh.makeCube()
    counter = 0
    for f in mesh.topology.getValidFaces():
        counter = counter + 1
    assert counter == mesh.topology.getValidFaces().count()
