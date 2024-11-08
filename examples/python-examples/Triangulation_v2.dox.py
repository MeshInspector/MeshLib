## \page ExamplePythonTriangulationV2 Simple triangulation V2
##
## Simple triangulation
##
## \code
from meshlib import mrmeshpy as mm
from meshlib import mrmeshnumpy as mn
import numpy as np

u, v = np.mgrid[0:2 * np.pi:100j, 0:np.pi:100j]
x = np.cos(u) * np.sin(v)
y = np.sin(u) * np.sin(v)
z = np.cos(v)

# Prepare for MeshLib PointCloud
verts = np.stack((x.flatten(), y.flatten(), z.flatten()), axis=-1).reshape(-1, 3)
# Create MeshLib PointCloud from np ndarray
# Create MeshLib PointCloud from np ndarray
pc = mn.pointCloudFromPoints(verts)
# Remove duplicate points
pc.validPoints = mm.pointUniformSampling(pc, 1e-3)
pc.invalidateCaches()

# Triangulate it
triangulated_pc = mm.triangulatePointCloud(pc)

# Fix possible issues
params = mm.OffsetParameters()
params.voxelSize = pc.computeBoundingBox().diagonal() * 5e-2
triangulated_pc = mm.offsetMesh(triangulated_pc, 0.1, params=params)

## \endcode
