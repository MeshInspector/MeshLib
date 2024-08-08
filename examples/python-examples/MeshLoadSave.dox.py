## \page ExamplePythonMeshLoadSave Loading and saving mesh files
##
## Load and save example:
##
## \code
import meshlib.mrmeshpy as mrmeshpy

try:
    mesh = mrmeshpy.loadMesh("mesh.stl")
except ValueError as e:
    print(e)

mrmeshpy.saveMesh(mesh, "mesh.ply")
## \endcode
