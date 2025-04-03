from meshlib import mrmeshpy

# Create some mesh
mesh = mrmeshpy.makeTorus(primaryRadius=1.0,
                          secondaryRadius=0.1,
                          primaryResolution=16,
                          secondaryResolution=16)
verts_num = mesh.points.size()

# Create VertScalars obj with weights for every vertice
scalars = mrmeshpy.VertScalars(verts_num)
for i in range(verts_num):
    weight = abs(mesh.points.vec[i].x / 5) + 0.2 # Offset size for point
    scalars.vec[i] = weight

# ===params

params = mrmeshpy.WeightedPointsShellParameters()
params.offset = 0.0 # common offset that will be added for every point to weighted
# Algorithm is voxel based, voxel size affects performance and form of result mesh
params.voxelSize = mrmeshpy.suggestVoxelSize(mesh, 10000)
params.dist.maxWeight = max(scalars.vec) # should always have maximum between weights provided

# ===offset
res = mrmeshpy.weightedMeshShell(mesh, scalars, params)

mrmeshpy.saveMesh(res, "offset_weighted.ctm")
