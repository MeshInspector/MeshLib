import * as ml from '@meshinspector/meshlib';

await ml.init();

// Load meshes
using mesh = ml.MeshLoad.fromAnySupportedFormat('meshAwithHole.stl');
using meshB = ml.MeshLoad.fromAnySupportedFormat('meshBwithHole.stl');

// Unite meshes
mesh.addMesh(meshB);

// Find holes (expect that there are exactly 2 holes)
using edges = mesh.topology.findHoleRepresentiveEdges();
if (edges.size() !== 2)
  throw new Error(`expected exactly 2 holes, found ${edges.size()}`);

// Connect two holes
using params = new ml.StitchHolesParams();
params.metric = ml.getUniversalMetric(mesh);
ml.stitchHoles(mesh, edges.at(0), edges.at(1), params);

// Save result
ml.MeshSave.toAnySupportedFormat(mesh, 'stitchedMesh.stl');
