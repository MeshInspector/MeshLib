import * as ml from '@meshinspector/meshlib';

await ml.init();

using mesh1 = ml.MeshLoad.fromAnySupportedFormat('mesh1.ctm');
using mesh2 = ml.MeshLoad.fromAnySupportedFormat('mesh2.ctm');

// the single minimal signed distance between two meshes
using result = ml.findSignedDistance(new ml.MeshPart(mesh1), new ml.MeshPart(mesh2));
console.log(`Signed distance between meshes is ${result.signedDist}`);
