import * as ml from '@meshinspector/meshlib';

await ml.init();

// load points
using pointCloud = ml.PointsLoad.fromAnySupportedFormat('Points.ply');

// triangulate (the result is empty if the triangulation was cancelled)
using triangulated = ml.triangulatePointCloud(pointCloud);
const mesh = triangulated.value();
if (!mesh)
  throw new Error('triangulation was cancelled');

ml.MeshSave.toAnySupportedFormat(mesh, 'Mesh.ctm');
