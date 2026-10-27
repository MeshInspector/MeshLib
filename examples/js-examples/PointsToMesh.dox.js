import * as ml from '@meshinspector/meshlib';

await ml.init();

// load points
using pointCloud = ml.PointsLoad.fromAnySupportedFormat('Points.ply');

// triangulate (the result is empty if the triangulation was cancelled)
using triangulated = ml.triangulatePointCloud(pointCloud);
if (!triangulated.has_value())
  throw new Error('triangulation was cancelled');

ml.MeshSave.toAnySupportedFormat(triangulated.value(), 'Mesh.ctm');
