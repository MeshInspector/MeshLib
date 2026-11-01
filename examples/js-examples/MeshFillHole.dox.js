import * as ml from '@meshinspector/meshlib';

await ml.init();

// Load mesh
using mesh = ml.MeshLoad.fromAnySupportedFormat('mesh.stl');

// Find single edge for each hole in mesh
using holeEdges = mesh.topology.findHoleRepresentiveEdges();

// Setup filling parameters
using params = new ml.FillHoleParams();
params.metric = ml.getUniversalMetric(mesh);

// Fill all holes at once (ml.fillHole(mesh, edge, params) fills a single hole)
ml.fillHoles(mesh, holeEdges, params);

// Save result
ml.MeshSave.toAnySupportedFormat(mesh, 'filledMesh.stl');
