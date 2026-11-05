import * as ml from '@meshinspector/meshlib';

await ml.init();

using mesh = ml.MeshLoad.fromAnySupportedFormat('mesh1.ctm');
const point = new ml.Vector3f(1.5, 1.0, 0.5);

using result = ml.findSignedDistance(point, new ml.MeshPart(mesh));
const found = result.value(); // null if nothing was found
if (found)
  console.log(`Signed distance from point to mesh ${found.dist}`);
