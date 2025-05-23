from meshlib import mrmeshpy

# Create some mesh
mesh = mrmeshpy.makeTorus(primaryRadius=1.0,
                          secondaryRadius=0.1,
                          primaryResolution=16,
                          secondaryResolution=16)
verts_num = mesh.points.size()

# Create VertScalars obj with weights for every vertex
scalars = mrmeshpy.VertScalars(verts_num)
for i in range(verts_num):
    weight = abs(mesh.points.vec[i].x / 5) # Individual extra offset sizes for points
    scalars.vec[i] = weight

# ===params

params = mrmeshpy.WeightedShell.ParametersMetric()
# Algorithm is voxel based, voxel size affects performance and form of result mesh
params.voxelSize = mrmeshpy.suggestVoxelSize(mesh, 10000)
# common basic offset applied for all point
# Vertex-specific weighted offsets applied after the basic one
params.offset = 0.2
params.dist.maxWeight = max(scalars.vec) # should always have maximum between weights provided

# ===offset
res = mrmeshpy.WeightedShell.meshShell(mesh, scalars, params)

mrmeshpy.saveMesh(res, "offset_weighted.ctm")
