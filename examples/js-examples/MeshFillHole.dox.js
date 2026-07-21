import createMeshLib from '@meshinspector/meshlib';

const ml = await createMeshLib();

// Load mesh
using mesh = ml.MeshLoad.fromAnySupportedFormat('mesh.stl');

// Find a single representative edge for each hole in the mesh
using topology = mesh.topology;
const holeEdges = topology.findHoleRepresentiveEdges();

// Setup filling parameters
using params = new ml.FillHoleParams();
using metric = ml.getUniversalMetric(mesh);
params.metric = metric;

// Fill all holes at once (ml.fillHole( mesh, edge, params ) fills a single hole)
ml.fillHoles(mesh, holeEdges, params);

// Save result
ml.MeshSave.toAnySupportedFormat(mesh, 'filledMesh.stl');
