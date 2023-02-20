from helper import *
import pytest
import shutil

def test_save_voxels():
    mesh = mm.makeTorus()
    mtvParams = mm.MeshToVolumeParams()
    mtvParams.type = mm.MeshToVolumeParamsType.Signed
    mtvParams.surfaceOffset = 3
    meshBox = mesh.computeBoundingBox()
    boxSize = meshBox.max-meshBox.min
    mtvParams.voxelSize = boxSize / 27.0
    voxels = mm.meshToVolume(mesh,mtvParams)

    vsParams = mm.VoxelsSaveSavingSettings()
    os.mkdir("save_voxels_dir_test")
    vsParams.path = "save_voxels_dir_test"
    vsParams.slicePlane = mm.SlicePlane.XY
    mm.saveAllSlicesToImage(voxels,vsParams)
    shutil.rmtree("save_voxels_dir_test")
