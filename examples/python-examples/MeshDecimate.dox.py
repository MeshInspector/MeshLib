## \page ExamplePythonMeshDecimate Mesh decimate
##
## Example of mesh decimate
##
## \code
import meshlib.mrmeshpy as mrmeshpy

# Load mesh
mesh = mrmeshpy.loadMesh("mesh.stl")

# Setup decimate parameters
settings = mrmeshpy.DecimateSettings()
settings.maxError = 0.05

# Decimate mesh
mrmeshpy.decimateMesh(mesh, settings)

# Save result
mrmeshpy.saveMesh(mesh, "decimatedMesh.stl")

## \endcode
