from helper import *
import pytest


def test_save_slice_to_image():
    mesh = mrmesh.makeCube(mrmesh.Vector3f(5.0, 5.0, 5.0))
    params = mrmesh.MeshToVolumeParams()
    params.type = mrmesh.MeshToVolumeParamsType.Unsigned
    params.voxelSize = mrmesh.Vector3f(0.1, 0.1, 0.1)
    try:
        vdbvolume = mrmesh.meshToVolume(mesh, params)
    except ValueError as e:
        print(e)
        assert False

    try:
        mrmesh.saveSliceToImage("slice.png", vdbvolume, mrmesh.SlicePlain.XY, 26)
    except ValueError as e:
        print(e)
        assert False
