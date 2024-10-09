## \page ExamplePythonViewer Viewer Example
##
## Example of using Viewer
##
## \code
import os
from meshlib import mrmeshpy as mm
from meshlib import mrviewerpy as mv

# Load mesh
mesh = mm.loadMesh("mesh.stl")

# Setup parameters
params = mm.OffsetParameters()
params.voxelSize = mesh.computeBoundingBox().diagonal() * 5e-3  # offset grid precision (algorithm is voxel based)
if mm.findRightBoundary(mesh.topology).empty():
    params.signDetectionMode = mm.SignDetectionMode.HoleWindingRule  # use if you have holes in mesh

# Make offset mesh
offset = mesh.computeBoundingBox().diagonal() * 0.05
result_mesh = mm.offsetMesh(mesh, offset, params)


mv.launch() # a window will be opened
mv.addMeshToScene(mesh, "Mesh 1") # show initial mesh
mv.Viewer().preciseFitDataViewport() # fit viewer to the mesh

# user can manipulate with viewer window while this python is on pause
os.system("pause")

# remove all objects from scene
mv.clearScene()

# add offset mesh to scene
mv.addMeshToScene(result_mesh, "Mesh Offset")
# user can manipulate with viewer window while this python is on pause
os.system("pause")

# close viewer window nicely
mv.Viewer().shutdown()

## \endcode