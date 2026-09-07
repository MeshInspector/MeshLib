import createMeshLib from '@meshinspector/meshlib';

const ml = await createMeshLib();

using mesh1 = ml.MeshLoad.fromAnySupportedFormat('mesh1.ply');
using mesh2 = ml.MeshLoad.fromAnySupportedFormat('mesh2.ply');

// unlike findSignedDistances, this returns the single minimal signed distance between two meshes
const dist = ml.findSignedDistanceFromMesh(mesh1, mesh2);
console.log(`Signed distance between meshes is ${dist.signedDist}`);
