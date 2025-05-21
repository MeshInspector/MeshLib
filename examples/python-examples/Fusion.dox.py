from meshlib import mrmeshpy as mm

points = mm.loadPoints("Points.ply")

params = mm.PointsToMeshParameters()
params.voxelSize = points.computeBoundingBox().diagonal()*1e-2
params.sigma = max(params.voxelSize,mm.findAvgPointsRadius(points,50))
params.minWeight = 1

mesh = mm.pointsToMeshFusion(points,params)

mm.saveMesh(mesh,"Mesh.ctm")
