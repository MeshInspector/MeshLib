import createMeshLib from '@meshinspector/meshlib';

const ml = await createMeshLib();

using mesh = ml.MeshLoad.fromAnySupportedFormat('mesh1.ply');
const point = { x: 1.5, y: 1.0, z: 0.5 };

const result = ml.findSignedDistanceFromPoint(point, mesh);
if (result)
  console.log(`Signed distance from point to mesh ${result.dist}`);
