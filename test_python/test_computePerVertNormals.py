import pytest
from helper import *
import meshlib.mrmeshnumpy as mrmeshnumpy

def test_compute_per_vert_normals():
    torus = mrmesh.makeTorus(2, 1, 10, 10, None)

    normals = mrmesh.computePerVertNormals(torus)
    normals1 = mrmesh.computePerVertPseudoNormals(torus)

    numpyNormals = mrmeshnumpy.toNumpyArray(normals)
