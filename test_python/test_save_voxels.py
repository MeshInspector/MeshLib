from helper import *
import pytest
import shutil

def test_save_voxels():
    mesh = mrmesh.makeTorus()
    mtvParams = mrmesh.MeshToVolumeParams()
    mtvParams.type = mrmesh.MeshToVolumeParamsType.Signed
    mtvParams.surfaceOffset = 3
    meshBox = mesh.computeBoundingBox()
    boxSize = meshBox.max-meshBox.min
    mtvParams.voxelSize = boxSize / 27.0
    voxels = mrmesh.meshToVolume(mesh,mtvParams)

    vsParams = mrmesh.VoxelsSaveSavingSettings()
    os.mkdir("save_voxels_dir_test")
    vsParams.path = "save_voxels_dir_test"
    vsParams.slicePlane = mrmesh.SlicePlane.XY
    mrmesh.saveAllSlicesToImage(voxels,vsParams)
    shutil.rmtree("save_voxels_dir_test")
