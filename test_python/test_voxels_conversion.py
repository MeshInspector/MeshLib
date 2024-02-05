import unittest as ut

import numpy as np
import pytest
import math
from helper import *
from meshlib import mrmeshpy
from meshlib import mrmeshnumpy


def test_voxels_conversion():
    # Create array containing one octant with spherical distribution
    mul = 1
    np_array = np.zeros((20 * mul, 30 * mul, 40 * mul))
    max_value = 3000 * mul * mul
    center = (np_array.shape[0] / 10, np_array.shape[1] / 3, np_array.shape[2] / 3)
    for k in range(np_array.shape[2]):
        for j in range(np_array.shape[1]):
            for i in range(np_array.shape[0]):
                x, y, z = i - center[0], j - center[1], k - center[2]
                np_array[i, j, k] = 0.0 if x <= 0 or y <= 0 or z <= 0 else \
                        max_value - (x * x + y * y + z * z)

    # Create initial volume object
    volume0 = mrmeshnumpy.simpleVolumeFrom3Darray(np_array)
    volume0.voxelSize = mrmeshpy.Vector3f(0.5, 1, 1)

    # Test conversion functions
    vdb_volume = mrmeshpy.simpleVolumeToVdbVolume(volume0)
    volume = mrmeshpy.vdbVolumeToSimpleVolume(vdb_volume)
    grid = mrmeshpy.simpleVolumeToDenseGrid(volume)
    # Basic results validation
    assert volume0.dims == vdb_volume.dims == volume.dims
    assert volume0.voxelSize == vdb_volume.voxelSize == volume.voxelSize
    # min/max not preserved
    # assert volume0.min == vdb_volume.min == volume.min
    # assert volume0.max == vdb_volume.max == volume.max

    # Test mesh build functions
    radius = 15 * mul
    mesh1 = mrmeshpy.gridToMesh(vdb_volume, isoValue = max_value - radius * radius)
    mesh2 = mrmeshpy.gridToMesh(grid, voxelSize = volume0.voxelSize, isoValue = max_value - radius * radius)
    # Test by roughly comparing mesh volume to expectation (volume of 1/8 sphere)
    vol = radius ** 3 * math.pi / 6 * (volume0.voxelSize.x * volume0.voxelSize.y * volume0.voxelSize.z)
    assert 0.7 < mesh1.volume() / vol < 1.3
    assert 0.7 < mesh2.volume() / vol < 1.3
