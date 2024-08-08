import unittest as ut

import numpy as np
import pytest
from helper import *
import_mrmeshnumpy()


def test_numpy_voxels():
    sphere = mrmesh.makeSphere(mrmesh.SphereParams())
    params = mrmesh.MeshToDistanceVolumeParams()
    voxelSize = 0.01
    box = sphere.computeBoundingBox()
    expansion = mrmesh.Vector3f.diagonal(3 * voxelSize)
    params.origin = box.min - expansion
    params.voxelSize = mrmesh.Vector3f.diagonal(0.01)
    dimensionsF = (box.max + expansion - params.origin) / voxelSize
    params.dimensions = mrmesh.Vector3i(
        int(dimensionsF.x), int(dimensionsF.y), int(dimensionsF.z)
    ) + mrmesh.Vector3i.diagonal(1)
    params.signMode = mrmesh.SignDetectionMode.HoleWindingRule
    params.maxDistSq = 3 * voxelSize
    volume = mrmesh.meshToDistanceVolume(sphere, params)
    npArray = mrmeshnumpy.getNumpy3Darray(volume)
    assert npArray.shape[0] == params.dimensions.x
    assert npArray.shape[1] == params.dimensions.y
    assert npArray.shape[2] == params.dimensions.z
