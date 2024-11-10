import meshlib.mrmeshpy as mrmeshpy

# Load meshes
meshFloating = mrmeshpy.loadMesh("meshA.stl")
meshFixed = mrmeshpy.loadMesh("meshB.stl")

# Prepare ICP parameters
diagonal = meshFixed.getBoundingBox().diagonal()
icp_sampling_voxel_size = diagonal * 0.01  # To sample points from object
icp_params = mrmeshpy.ICPProperties()
icp_params.distThresholdSq = (diagonal * 0.1) ** 2  # Use points pairs with maximum distance specified
icp_params.exitVal = diagonal * 0.003  # Stop when this distance reached

# Calculate transformation
icp = mrmeshpy.ICP(meshFloating, meshFixed,
                   mrmeshpy.AffineXf3f(), mrmeshpy.AffineXf3f(),
                   icp_sampling_voxel_size)
icp.setParams(icp_params)
xf = icp.calculateTransformation()

# Transform floating mesh
meshFloating.transform(xf)

# Output information string
print(icp.getLastICPInfo())

# Save result
mrmeshpy.saveMesh(meshFloating, "meshA_icp.stl")
