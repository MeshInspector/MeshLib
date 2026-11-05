import * as ml from '@meshinspector/meshlib';

await ml.init();

// Load mesh
using mesh = ml.MeshLoad.fromAnySupportedFormat('mesh.stl');

// you can set various parameters for the resolving process; see the documentation for more info
using params = new ml.FixMeshDegeneraciesParams();
params.maxDeviation = 1e-5 * mesh.computeBoundingBox().diagonal();
params.tinyEdgeLength = 1e-3;

ml.fixMeshDegeneracies(mesh, params);

// Save result
ml.MeshSave.toAnySupportedFormat(mesh, 'fixed_mesh.stl');
