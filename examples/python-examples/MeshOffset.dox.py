import meshlib.mrmeshpy as mrmeshpy
import sys

# Create mesh
mesh = mrmeshpy.makeCube()

# Setup parameters
params = mrmeshpy.OffsetParameters()
# calculate voxel size depending on desired accuracy and/or memory consumption
params.voxelSize = mrmeshpy.suggestVoxelSize(mesh, 10000000)
if mrmeshpy.findRightBoundary(mesh.topology).empty():
    params.signDetectionMode = mrmeshpy.SignDetectionMode.HoleWindingRule  # use if you have holes in mesh

# Make offset mesh
offset = mesh.computeBoundingBox().diagonal() * 0.1
try:
    result_mesh = mrmeshpy.offsetMesh(mesh, offset, params)
except ValueError as e:
    print(e)
    sys.exit(1)


# Save result
mrmeshpy.saveMesh(result_mesh, "mesh_offset.stl")
