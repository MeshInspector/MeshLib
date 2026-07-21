import createMeshLib from '@meshinspector/meshlib';

const ml = await createMeshLib();

using refMesh = ml.MeshLoad.fromAnySupportedFormat('mesh1.ply');
using mesh = ml.MeshLoad.fromAnySupportedFormat('mesh2.ply');

// distances between the points of the target mesh and the reference mesh
// (the JS binding requires a parameters object; its defaults match the C++ overload)
using params = new ml.MeshProjectionParameters();
const vertDistances = ml.findSignedDistances(refMesh, mesh, params);

// find the closest and farthest distances (a loop, not Math.min(...arr), because the
// typed array may be large enough to overflow the call stack when spread)
let closest = Infinity;
let farthest = -Infinity;
for (const d of vertDistances) {
  if (d < closest) closest = d;
  if (d > farthest) farthest = d;
}
console.log(`Distance between reference mesh and the closest point of target mesh is ${closest}`);
console.log(`Distance between reference mesh and the farthest point of target mesh is ${farthest}`);
