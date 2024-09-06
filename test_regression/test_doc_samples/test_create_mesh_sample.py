import pytest

from helpers.common_helpers import list_compare_2d
from module_helper import *


@pytest.mark.smoke
def test_create_mesh_get_primitives_sample():
    # === Code sample

    import numpy as np

    faces = np.ndarray(shape=(2, 3), dtype=np.int32, buffer=np.array([[0, 1, 2], [2, 3, 0]], dtype=np.int32))

    # mrmeshpy uses float32 for vertex coordinates
    # however, you could also use float64
    verts = np.ndarray(shape=(4, 3), dtype=np.float32,
                       buffer=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
                                       dtype=np.float32))

    mesh = mrmeshnumpy.meshFromFacesVerts(faces, verts)

    # some mesh manipulations

    outVerts = mrmeshnumpy.getNumpyVerts(mesh)
    outFaces = mrmeshnumpy.getNumpyFaces(mesh.topology)

    # === verification

    assert list_compare_2d(outVerts, [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    assert list_compare_2d(outFaces, [[0, 1, 2], [2, 3, 0]])

    # another option
    outVerts1 = mrmeshnumpy.toNumpyArray(mesh.points)
    assert list_compare_2d(outVerts1, [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
