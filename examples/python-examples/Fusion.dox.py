from meshlib import mrmeshpy as mm

points = mm.loadPoints("NefertitiPoints.ply")

params = mm.PointsToMeshParameters()
params.voxelSize = points.computeBoundingBox().diagonal()*2e-3
params.sigma = max(params.voxelSize,mm.findAvgPointsRadius(points,40))
params.minWeight = 1

mesh = mm.pointsToMeshFusion(points,params)

mm.saveMesh(mesh,"NefertitiMesh.ply")
