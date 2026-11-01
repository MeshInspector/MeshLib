import * as ml from '@meshinspector/meshlib';

await ml.init();

using refMesh = ml.MeshLoad.fromAnySupportedFormat('mesh1.ctm');
using mesh = ml.MeshLoad.fromAnySupportedFormat('mesh2.ctm');

// distances between the points of the target mesh and the reference mesh, as a Float32Array
using vertDistances = ml.findSignedDistances(refMesh, mesh);
const distances = vertDistances.toArray();

// find the closest and farthest distances (a loop, not Math.min(...arr), because the
// typed array may be large enough to overflow the call stack when spread)
let closest = Infinity;
let farthest = -Infinity;
for (const d of distances) {
  if (d < closest) closest = d;
  if (d > farthest) farthest = d;
}
console.log(`Distance between reference mesh and the closest point of target mesh is ${closest}`);
console.log(`Distance between reference mesh and the farthest point of target mesh is ${farthest}`);
