import os
import sys

from meshlib import mrmeshpy as mm
from meshlib import mrviewerpy as mv

# This example needs an input mesh; create one if you do not have it already
if not os.path.exists("mesh.stl"):
    mm.saveMesh(mm.makeCube(), "mesh.stl")

# The Viewer needs a graphical session. On a headless Linux machine (container, CI,
# WSL without an X server, SSH without X forwarding) no window can be opened, and in
# releases before 3.1.3.566 the failure was silent and the next viewer call hung forever.
if sys.platform.startswith("linux") and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
    sys.exit("MeshLib Viewer requires a graphical session, but no display was found "
             "(neither DISPLAY nor WAYLAND_DISPLAY is set). "
             "On a headless machine start with the mesh loading and saving example "
             "(MeshLoadSave.dox.py) instead.")

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


# Open a window; raises where the Viewer is unavailable, e.g. on macOS
try:
    mv.launch()
except RuntimeError as e:
    sys.exit(f"Could not start MeshLib Viewer: {e}")

mv.addMeshToScene(mesh, "Mesh 1") # show initial mesh
mv.Viewer().preciseFitDataViewport() # fit viewer to the mesh
mv.selectByName("Mesh 1")

mv.Viewer().preciseFitDataViewport() # fit viewer to the mesh
mv.Viewer().showSceneTree(True) # enables Scene Tree in Viewer window
# user can manipulate with viewer window while this python is on pause
input("Press Enter to continue...")

# remove all objects from scene
mv.clearScene()

# add offset mesh to scene
mv.addMeshToScene(result_mesh, "Mesh Offset")
mv.selectByName("Mesh Offset")
mv.Viewer().showSceneTree(False) # disables Scene Tree in Viewer window
# user can manipulate with viewer window while this python is on pause
input("Press Enter to continue...")

# close viewer window nicely
mv.Viewer().shutdown()
