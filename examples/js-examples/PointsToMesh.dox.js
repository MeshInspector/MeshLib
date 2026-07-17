import createMeshLib from '@meshinspector/meshlib';

const ml = await createMeshLib();

// load points
using pointCloud = ml.PointsLoad.fromAnySupportedFormat('Points.ply');

// triangulate (the JS binding requires a parameters object)
using params = new ml.TriangulationParameters();
using mesh = ml.triangulatePointCloud(pointCloud, params);
if (!mesh)
  throw new Error('triangulation was cancelled');

ml.MeshSave.toAnySupportedFormat(mesh, 'Mesh.ply');
