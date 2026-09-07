import createMeshLib from '@meshinspector/meshlib';

const ml = await createMeshLib();

using pointCloud = ml.PointsLoad.fromAnySupportedFormat('Points.ply');

using bbox = pointCloud.computeBoundingBox();
using params = new ml.PointsToMeshParameters();
params.voxelSize = bbox.diagonal() * 0.01;
params.sigma = Math.max(params.voxelSize, ml.findAvgPointsRadius(pointCloud, 50));
params.minWeight = 1;

using mesh = ml.pointsToMeshFusion(pointCloud, params);

ml.MeshSave.toAnySupportedFormat(mesh, 'Mesh.ply');
