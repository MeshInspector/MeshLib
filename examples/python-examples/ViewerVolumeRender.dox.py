import sys
from meshlib import mrmeshpy as mm
from meshlib import mrviewerpy as mv
import os

# load volume file
volume = mm.loadVoxels("stag_beetle.dcm")[0]

# setup scene object for voxels data
ov = mm.ObjectVoxels()
ov.setName("Beetle")
ov.construct(volume)
ov.setIsoValue(volume.min * 0.6 + volume.max * 0.4)
ov.select(True)

# add it to scene
mm.SceneRoot.get().addChild(ov)

# start viewer
mv.launch(mv.ViewerLaunchParams(),mv.ViewerSetup())

# enable volume rendering (important to do in GUI thread)
mv.runFromGUIThread( lambda : ov.enableVolumeRendering(True) )

mv.Viewer().preciseFitDataViewport()

# setup volume rendering settings
vrp = mm.ObjectVoxels.VolumeRenderingParams()
vrp.alphaType = mm.ObjectVoxels.VolumeRenderingParams.AlphaType.LinearIncreasing
vrp.min = volume.min
vrp.max = volume.max
vrp.alphaLimit = 150
vrp.lutType = mm.ObjectVoxels.VolumeRenderingParams.LutType.Rainbow
vrp.shadingType = mm.ObjectVoxels.VolumeRenderingParams.ShadingType.ValueGradient

# apply volume rendering settings in GUI thread
mv.runFromGUIThread( lambda : ov.setVolumeRenderingParams(vrp) )

# fit camera
mv.Viewer().preciseFitDataViewport()

os.system("pause")