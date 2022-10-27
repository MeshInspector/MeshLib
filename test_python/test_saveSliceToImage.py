from helper import *
import pytest


def test_save_slice_to_image():
    mesh = mrmesh.makeCube(mrmesh.Vector3f(5.0, 5.0, 5.0))
    params = mrmesh.MeshToVolumeParams()
    params.type = mrmesh.MeshToVolumeParamsType.Unsigned
    params.voxelSize = mrmesh.Vector3f(0.1, 0.1, 0.1)
    expvdbvolume = mrmesh.meshToVolume(mesh, params)
    assert (expvdbvolume.has_value())
    assert (mrmesh.saveSliceToImage("slice.png", expvdbvolume.value(), mrmesh.SlicePlain.XY, 26).has_value())
