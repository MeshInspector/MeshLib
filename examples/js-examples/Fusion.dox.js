import * as ml from '@meshinspector/meshlib';

await ml.init();

using pointCloud = ml.PointsLoad.fromAnySupportedFormat('Points.ply');

using params = new ml.PointsToMeshParameters();
params.voxelSize = pointCloud.computeBoundingBox().diagonal() * 1e-2;
params.sigma = Math.max(params.voxelSize, ml.findAvgPointsRadius(pointCloud, 50));
params.minWeight = 1;

using mesh = ml.pointsToMeshFusion(pointCloud, params);
ml.MeshSave.toAnySupportedFormat(mesh, 'Mesh.ctm');
