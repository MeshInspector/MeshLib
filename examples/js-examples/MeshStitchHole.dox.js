import createMeshLib from '@meshinspector/meshlib';

const ml = await createMeshLib();

// Load meshes
using mesh = ml.MeshLoad.fromAnySupportedFormat('meshAwithHole.stl');
using meshB = ml.MeshLoad.fromAnySupportedFormat('meshBwithHole.stl');

// Unite meshes
mesh.addMesh(meshB);

// Find holes (expect that there are exactly 2 holes)
using topology = mesh.topology;
const edges = topology.findHoleRepresentiveEdges();
if (edges.length !== 2)
  throw new Error(`expected exactly 2 holes, found ${edges.length}`);

// Connect two holes
using params = new ml.StitchHolesParams();
using metric = ml.getUniversalMetric(mesh);
params.metric = metric;
ml.stitchHoles(mesh, edges[0], edges[1], params);

// Save result
ml.MeshSave.toAnySupportedFormat(mesh, 'stitchedMesh.stl');
